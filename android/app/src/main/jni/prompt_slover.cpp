#include "prompt_slover.h"

int PromptSlover::load(AAssetManager* mgr, std::string vocab)
{
	ncnn::set_cpu_powersave(0);
	ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

	net.opt.lightmode = true;
	net.opt.use_vulkan_compute = false;
	net.opt.use_winograd_convolution = false;
	net.opt.use_sgemm_convolution = false;
	net.opt.use_fp16_packed = true;
	net.opt.use_fp16_storage = true;
	net.opt.use_fp16_arithmetic = true;
	net.opt.use_packing_layout = true;
	net.opt.num_threads = ncnn::get_big_cpu_count();

	net.load_param(mgr,"FrozenCLIPEmbedder-fp16.param");
	net.load_model(mgr,"FrozenCLIPEmbedder-fp16.bin");

	std::ifstream infile;
	infile.open(vocab.data());
	std::string s;
	int idx = 0;
	while (getline(infile, s)) {
		tokenizer_token2idx.insert(pair<string, int>(s, idx));
		tokenizer_idx2token.insert(pair<int, string>(idx, s));
		idx++;
	}
	infile.close();

	return 0;
}

ncnn::Mat PromptSlover::get_conditioning(string& prompt)
{
	vector<pair<string, float>> parsed = parse_prompt_attention(prompt);

	// tokenתids
	vector<vector<int>> tokenized;
	{
		for (auto p : parsed) {
			vector<string> tokens = split(p.first);
			vector<int> ids;
			for (string token : tokens) {
				ids.push_back(tokenizer_token2idx[token]);
			}
			tokenized.push_back(ids);
		}
	}

	vector<int> remade_tokens;
	vector<float> multipliers;
	{
		int last_comma = -1;
		for (int it_tokenized = 0; it_tokenized < tokenized.size(); it_tokenized++) {
			vector<int> tokens = tokenized[it_tokenized];
			float weight = parsed[it_tokenized].second;

			int i = 0;
			while (i < tokens.size()) {
				int token = tokens[i];
				if (token == 267) {
					last_comma = remade_tokens.size();
				}
				else if ((max(int(remade_tokens.size()), 1) % 75 == 0) && (last_comma != -1) && (remade_tokens.size() - last_comma <= 20)) {
					last_comma += 1;
					vector<int> reloc_tokens(remade_tokens.begin() + last_comma, remade_tokens.end());
					vector<float> reloc_mults(multipliers.begin() + last_comma, multipliers.end());
					vector<int> _remade_tokens_(remade_tokens.begin(), remade_tokens.begin() + last_comma);
					remade_tokens = _remade_tokens_;
					int length = remade_tokens.size();
					int rem = ceil(length / 75.0) * 75 - length;
					vector<int> tmp_token(rem, 49407);
					remade_tokens.insert(remade_tokens.end(), tmp_token.begin(), tmp_token.end());
					remade_tokens.insert(remade_tokens.end(), reloc_tokens.begin(), reloc_tokens.end());
					vector<float> _multipliers_(multipliers.begin(), multipliers.end() + last_comma);
					vector<int> tmp_multipliers(rem, 1.0f);
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
		vector<int> tmp_token(tokens_to_add, 49407);
		remade_tokens.insert(remade_tokens.end(), tmp_token.begin(), tmp_token.end());
		vector<int> tmp_multipliers(tokens_to_add, 1.0f);
		multipliers.insert(multipliers.end(), tmp_multipliers.begin(), tmp_multipliers.end());
	}

	ncnn::Mat conds(768, 0);
	{
		while (remade_tokens.size() > 0) {
			vector<int> rem_tokens(remade_tokens.begin() + 75, remade_tokens.end());
			vector<float> rem_multipliers(multipliers.begin() + 75, multipliers.end());

			vector<int> current_tokens;
			vector<float> current_multipliers;
			if (remade_tokens.size() > 0) {
				current_tokens.insert(current_tokens.end(), remade_tokens.begin(), remade_tokens.begin() + 75);
				current_multipliers.insert(current_multipliers.end(), multipliers.begin(), multipliers.begin() + 75);
			}
			else {
				vector<int> tmp_token(75, 49407);
				current_tokens.insert(current_tokens.end(), tmp_token.begin(), tmp_token.end());
				vector<int> tmp_multipliers(75, 1.0f);
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

vector<pair<string, float>> PromptSlover::parse_prompt_attention(string& texts)
{
	vector<pair<string, float>> res;
	stack<int> round_brackets;
	stack<int> square_brackets;
	const float round_bracket_multiplier = 1.1;
	const float square_bracket_multiplier = 1 / 1.1;

	vector<string> ms;
	for (char c : texts) {
		string s = string(1, c);
		if (s == "(" || s == "[" || s == ")" || s == "]") {
			ms.push_back(s);
		}
		else {
			if (ms.size() < 1)
				ms.push_back("");
			string last = ms[ms.size() - 1];
			if (last == "(" || last == "[" || last == ")" || last == "]") {
				ms.push_back("");
			}
			ms[ms.size() - 1] += s;
		}
	}

	for (string text : ms) {
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

string PromptSlover::whitespace_clean(string& text)
{
	return regex_replace(text, regex("\\s+"), " ");
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
			string pat = string(1, str[pos]);
			if (s.length() > 0)
				result.push_back(s + "</w>");
			if (pat != " ")
				result.push_back(pat + "</w>");
			i = pos;
		}
	}
	return result;
}