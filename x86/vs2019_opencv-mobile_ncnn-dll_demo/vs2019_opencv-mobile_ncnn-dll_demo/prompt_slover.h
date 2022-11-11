#pragma once
#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <stack>
#include <fstream>
#include <map>
#include <math.h>
#include <net.h>
using namespace std;

class PromptSlover
{
public:
    PromptSlover();

    ncnn::Mat get_conditioning(string& prompt);

private:
    std::vector<std::string> split(std::string str);
    string whitespace_clean(string& text);
    vector<pair<string, float>> parse_prompt_attention(string& texts);

    map<string, int> tokenizer_token2idx;
    map<int, string> tokenizer_idx2token;

    ncnn::Net net;
};