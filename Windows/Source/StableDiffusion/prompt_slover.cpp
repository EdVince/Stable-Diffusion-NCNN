#include "prompt_slover.h"

PromptSlover::PromptSlover()
{
	// 加载CLIP模型
	net.opt.use_vulkan_compute = false;
	net.opt.use_winograd_convolution = false;
	net.opt.use_sgemm_convolution = false;
	net.opt.use_fp16_packed = true;
	net.opt.use_fp16_storage = true;
	net.opt.use_fp16_arithmetic = true;
	net.opt.use_packing_layout = true;
	net.load_param("assets/FrozenCLIPEmbedder-fp16.param");
	net.load_model("assets/FrozenCLIPEmbedder-fp16.bin");

	// 读取tokenizer字典
	std::ifstream infile;
	std::string pathname = "assets/vocab.txt";
	infile.open(pathname.data());
	std::string s;
	int idx = 0;
	while (getline(infile, s)) {
		tokenizer_token2idx.insert(std::pair<std::string, int>(s, idx));
		tokenizer_idx2token.insert(std::pair<int, std::string>(idx, s));
		idx++;
	}
	infile.close();
}

ncnn::Mat PromptSlover::get_conditioning(std::string& prompt)
{
	// 重要度计算可以匹配“()”和“[]”，圆括号是加重要度，方括号是减重要度
    std::vector<std::pair<std::string, float>> parsed = parse_prompt_attention(prompt);

	// token转ids
    std::vector<std::vector<int>> tokenized;
	{
		for (auto p : parsed) {
            std::vector<std::string> tokens = split(p.first);
            std::vector<int> ids;
			for (std::string token : tokens) {
				ids.push_back(tokenizer_token2idx[token]);
			}
			tokenized.push_back(ids);
		}
	}

	// 一些处理
    std::vector<int> remade_tokens;
    std::vector<float> multipliers;
	{
		int last_comma = -1;
		for (int it_tokenized = 0; it_tokenized < tokenized.size(); it_tokenized++) {
            std::vector<int> tokens = tokenized[it_tokenized];
			float weight = parsed[it_tokenized].second;

			int i = 0;
			while (i < tokens.size()) {
				int token = tokens[i];
				if (token == 267) {
					last_comma = remade_tokens.size();
				}
				else if ((max(int(remade_tokens.size()), 1) % 75 == 0) && (last_comma != -1) && (remade_tokens.size() - last_comma <= 20)) {
					last_comma += 1;
                    std::vector<int> reloc_tokens(remade_tokens.begin() + last_comma, remade_tokens.end());
                    std::vector<float> reloc_mults(multipliers.begin() + last_comma, multipliers.end());
                    std::vector<int> _remade_tokens_(remade_tokens.begin(), remade_tokens.begin() + last_comma);
					remade_tokens = _remade_tokens_;
					int length = remade_tokens.size();
					int rem = ceil(length / 75.0) * 75 - length;
                    std::vector<int> tmp_token(rem, 49407);
					remade_tokens.insert(remade_tokens.end(), tmp_token.begin(), tmp_token.end());
					remade_tokens.insert(remade_tokens.end(), reloc_tokens.begin(), reloc_tokens.end());
                    std::vector<float> _multipliers_(multipliers.begin(), multipliers.end() + last_comma);
                    std::vector<int> tmp_multipliers(rem, 1.0f);
					_multipliers_.insert(_multipliers_.end(), tmp_multipliers.begin(), tmp_multipliers.end());
					_multipliers_.insert(_multipliers_.end(), reloc_mults.begin(), reloc_mults.end());
					multipliers = _multipliers_;
				}
				remade_tokens.push_back(token);
				multipliers.push_back(weight);
				i += 1;
			}
		}
		int prompt_target_length = ceil(max(int(remade_tokens.size()), 1) / 75.0) * 75;
		int tokens_to_add = prompt_target_length - remade_tokens.size();
        std::vector<int> tmp_token(tokens_to_add, 49407);
		remade_tokens.insert(remade_tokens.end(), tmp_token.begin(), tmp_token.end());
        std::vector<int> tmp_multipliers(tokens_to_add, 1.0f);
		multipliers.insert(multipliers.end(), tmp_multipliers.begin(), tmp_multipliers.end());
	}

	// 切分
	ncnn::Mat conds(768, 0);
	{
		while (remade_tokens.size() > 0) {
            std::vector<int> rem_tokens(remade_tokens.begin() + 75, remade_tokens.end());
            std::vector<float> rem_multipliers(multipliers.begin() + 75, multipliers.end());

            std::vector<int> current_tokens;
            std::vector<float> current_multipliers;
			if (remade_tokens.size() > 0) {
				current_tokens.insert(current_tokens.end(), remade_tokens.begin(), remade_tokens.begin() + 75);
				current_multipliers.insert(current_multipliers.end(), multipliers.begin(), multipliers.begin() + 75);
			}
			else {
                std::vector<int> tmp_token(75, 49407);
				current_tokens.insert(current_tokens.end(), tmp_token.begin(), tmp_token.end());
                std::vector<int> tmp_multipliers(75, 1.0f);
				current_multipliers.insert(current_multipliers.end(), tmp_multipliers.begin(), tmp_multipliers.end());
			}

			{
				ncnn::Mat token_mat = ncnn::Mat(77);
				token_mat.fill(int(49406));
				ncnn::Mat multiplier_mat = ncnn::Mat(77);
				multiplier_mat.fill(1.0f);

				int* token_ptr = token_mat;
				float* multiplier_ptr = multiplier_mat;
				for (int i = 0; i < 75; i++) {
					token_ptr[i + 1] = int(current_tokens[i]);
					multiplier_ptr[i + 1] = current_multipliers[i];
				}

				ncnn::Extractor ex = net.create_extractor();
				ex.set_light_mode(true);
				ex.input("token", token_mat);
				ex.input("multiplier", multiplier_mat);
				ex.input("cond", conds);
				ncnn::Mat new_conds;
				ex.extract("conds", new_conds);
				conds = new_conds;

			}

			remade_tokens = rem_tokens;
			multipliers = rem_multipliers;
		}
	}

	return conds;
}

std::vector<std::pair<std::string, float>> PromptSlover::parse_prompt_attention(std::string& texts)
{
    std::vector<std::pair<std::string, float>> res;
    std::stack<int> round_brackets;
    std::stack<int> square_brackets;
	const float round_bracket_multiplier = 1.1;
	const float square_bracket_multiplier = 1 / 1.1;

    std::vector<std::string> ms;
	for (char c : texts) {
        std::string s = std::string(1, c);
		if (s == "(" || s == "[" || s == ")" || s == "]") {
			ms.push_back(s);
		}
		else {
			if (ms.size() < 1)
				ms.push_back("");
            std::string last = ms[ms.size() - 1];
			if (last == "(" || last == "[" || last == ")" || last == "]") {
				ms.push_back("");
			}
			ms[ms.size() - 1] += s;
		}
	}

	for (std::string text : ms) {
		if (text == "(") {
			round_brackets.push(res.size());
		}
		else if (text == "[") {
			square_brackets.push(res.size());
		}
		else if (text == ")" && round_brackets.size() > 0) {
			for (int p = round_brackets.top(); p < res.size(); p++) {
				res[p].second *= round_bracket_multiplier;
			}
			round_brackets.pop();
		}
		else if (text == "]" and square_brackets.size() > 0) {
			for (int p = square_brackets.top(); p < res.size(); p++) {
				res[p].second *= square_bracket_multiplier;
			}
			square_brackets.pop();
		}
		else {
			res.push_back(make_pair(text, 1.0));
		}
	}

	while (!round_brackets.empty()) {
		for (int p = round_brackets.top(); p < res.size(); p++) {
			res[p].second *= round_bracket_multiplier;
		}
		round_brackets.pop();
	}

	while (!square_brackets.empty()) {
		for (int p = square_brackets.top(); p < res.size(); p++) {
			res[p].second *= square_bracket_multiplier;
		}
		square_brackets.pop();
	}

	int i = 0;
	while (i + 1 < res.size()) {
		if (res[i].second == res[i + 1].second) {
			res[i].first += res[i + 1].first;
			auto it = res.begin();
			res.erase(it + i + 1);
		}
		else {
			i += 1;
		}
	}

	return res;
}

std::string PromptSlover::whitespace_clean(std::string& text)
{
	return std::regex_replace(text, std::regex("\\s+"), " ");
}

std::vector<std::string> PromptSlover::split(std::string str)
{
	std::string::size_type pos;
	std::vector<std::string> result;
	str += " ";
	int size = str.size();
	for (int i = 0; i < size; i++)
	{
		pos = min(str.find(" ", i), str.find(",", i));
		if (pos < size)
		{
			std::string s = str.substr(i, pos - i);
            std::string pat = std::string(1, str[pos]);
			if (s.length() > 0)
				result.push_back(s + "</w>");
			if (pat != " ")
				result.push_back(pat + "</w>");
			i = pos;
		}
	}
	return result;
}
