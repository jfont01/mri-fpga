#include "config_loader.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

using std::string;
using std::unordered_map;

static inline string trim(const string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

unordered_map<string, string> load_conf_kv(const string& conf_path) {
    std::ifstream fin(conf_path);
    if (!fin) {
        throw std::runtime_error("No se pudo abrir conf: " + conf_path);
    }

    unordered_map<string, string> kv;
    string line;

    while (std::getline(fin, line)) {
        line = trim(line);
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        size_t eq = line.find('=');
        if (eq == string::npos) continue;

        string key = trim(line.substr(0, eq));
        string val = trim(line.substr(eq + 1));

        if (!val.empty() && ((val.front() == '"' && val.back() == '"') ||
                             (val.front() == '\'' && val.back() == '\''))) {
            val = val.substr(1, val.size() - 2);
        }

        kv[key] = val;
    }

    return kv;
}

string get_env(const char* name) {
    const char* p = std::getenv(name);
    return string(p);
}

string get_kv(const unordered_map<string, string>& kv, const string& key) {
    auto it = kv.find(key);
    return it->second;
}

string get_case_dir(){
    string GLOBAL_CONF_PATH    = get_env("GLOBAL_CONF_PATH");
    string PY_RUNNER           = get_env("PY_RUNNER");

    auto kv = load_conf_kv(GLOBAL_CONF_PATH);

    string N       = get_kv(kv, "N");
    string AF      = get_kv(kv, "AF");
    string L       = get_kv(kv, "L");
    string AXIS    = get_kv(kv, "AXIS");
    string PHANTOM = get_kv(kv, "PHANTOM");

    string case_dir = PY_RUNNER + "/output" + "/N" + N + "_Af" + AF + "_L" + L + "_axis" + AXIS + "_" + PHANTOM + "/gen";

    return case_dir;
}

string get_S_npy_path(){
    string case_dir             = get_case_dir();
    string sens_maps_dir        = case_dir + "/sens-maps";

    string S_path               = sens_maps_dir + "/smap.npy";

    return S_path;
}

string get_y_npy_path(){
    string case_dir             = get_case_dir();
    string coils_aliased_dir = case_dir + "/coils-aliased";

    string y_path = coils_aliased_dir + "/coil_aliased.npy";

    return y_path;
}

string get_A_dat_path(){
    string VM_ROOT              = get_env("VM_ROOT");
    string A_dat_path           = VM_ROOT + "/A/cpp_A.dat";

    return A_dat_path;
}

