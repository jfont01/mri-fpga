#ifndef _CONFIG_LOADER_H
#define _CONFIG_LOADER_H

#include <string>
#include <unordered_map>

using std::string;
using std::unordered_map;

unordered_map<string, string>   load_conf_kv(const string& conf_path);
string                          get_env(const char* name);
string                          get_kv(const unordered_map<string, string>& kv, const string& key);
string                          get_case_dir();
string                          get_S_npy_path();
string                          get_y_npy_path();
string                          get_A_dat_path();


#endif