#pragma once
#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <stack>
#include <fstream>
#include <map>
#include <math.h>
#include <ncnn/net.h>

class PromptSlover
{
public:
    PromptSlover();

    ncnn::Mat get_conditioning(std::string& prompt);

private:
    std::vector<std::string> split(std::string str);
    std::string whitespace_clean(std::string& text);
    std::vector<std::pair<std::string, float>> parse_prompt_attention(std::string& texts);

    std::map<std::string, int> tokenizer_token2idx;
    std::map<int, std::string> tokenizer_idx2token;

    ncnn::Net net;
};
