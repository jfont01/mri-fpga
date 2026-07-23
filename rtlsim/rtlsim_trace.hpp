#pragma once

#include "rtlsim_core.hpp"

#include <ap_fixed.h>

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rtl {

/*
 * --------------------------------------------------------------------------
 * Generic value formatting
 * --------------------------------------------------------------------------
 *
 * CSV:
 *   Human-readable representation.
 *
 * DAT:
 *   Bit-accurate comparison representation.
 *
 * Policy:
 *   bool / bit_t      -> 0 or 1
 *   ap_fixed<W,...>    -> signed raw integer in decimal
 *   ap_ufixed<W,...>   -> unsigned raw integer in decimal
 *   ap_int<W>          -> signed decimal
 *   ap_uint<W>         -> unsigned decimal
 */

template <typename T>
std::string value_to_string(const T& value)
{
    std::ostringstream oss;
    oss << std::boolalpha << value;
    return oss.str();
}

inline std::string value_to_dat(bool value)
{
    return value ? "1" : "0";
}

inline std::string value_to_dat(const bit_t& value)
{
    return value.to_bool() ? "1" : "0";
}

template <int W>
std::string value_to_dat(const ap_int<W>& value)
{
    return value.to_string(10);
}

template <int W>
std::string value_to_dat(const ap_uint<W>& value)
{
    return value.to_string(10);
}

template <
    int W,
    int I,
    ap_q_mode Q,
    ap_o_mode O,
    int N
>
std::string value_to_dat(const ap_fixed<W, I, Q, O, N>& value)
{
    ap_int<W> raw;
    raw.range(W - 1, 0) = value.range(W - 1, 0);

    return raw.to_string(10);
}

template <
    int W,
    int I,
    ap_q_mode Q,
    ap_o_mode O,
    int N
>
std::string value_to_dat(const ap_ufixed<W, I, Q, O, N>& value)
{
    ap_uint<W> raw;
    raw.range(W - 1, 0) = value.range(W - 1, 0);

    return raw.to_string(10);
}

template <typename T>
std::string value_to_dat(const T& value)
{
    return value_to_string(value);
}

/*
 * --------------------------------------------------------------------------
 * CSV utilities
 * --------------------------------------------------------------------------
 */

inline std::string csv_escape(const std::string& value)
{
    const bool needs_quotes =
        value.find(',')  != std::string::npos ||
        value.find('"')  != std::string::npos ||
        value.find('\n') != std::string::npos ||
        value.find('\r') != std::string::npos;

    if (!needs_quotes) {
        return value;
    }

    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');

    for (char c : value) {
        if (c == '"') {
            out.push_back('"');
            out.push_back('"');
        }
        else {
            out.push_back(c);
        }
    }

    out.push_back('"');
    return out;
}

/*
 * --------------------------------------------------------------------------
 * CsvTrace
 * --------------------------------------------------------------------------
 */

class CsvTrace {
public:
    using Reader = std::function<std::string()>;

    CsvTrace() = default;

    explicit CsvTrace(const std::string& path)
    {
        open(path);
    }

    void open(const std::string& path)
    {
        const std::filesystem::path file_path(path);
        const std::filesystem::path parent = file_path.parent_path();

        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }

        file_.open(file_path);

        if (!file_) {
            throw std::runtime_error("could not open trace file: " + file_path.string());
        }

        header_written_ = false;
    }

    void add_value(const std::string& name, Reader reader)
    {
        if (header_written_) {
            throw std::runtime_error("cannot add signal after CSV header was written");
        }

        names_.push_back(name);
        readers_.push_back(reader);
    }

    void sample(std::uint64_t cycle)
    {
        if (!file_) {
            return;
        }

        if (!header_written_) {
            write_header();
        }

        file_ << cycle;

        for (const Reader& reader : readers_) {
            file_ << "," << csv_escape(reader());
        }

        file_ << "\n";
    }

private:
    void write_header()
    {
        file_ << "cycle";

        for (const std::string& name : names_) {
            file_ << "," << csv_escape(name);
        }

        file_ << "\n";
        header_written_ = true;
    }

    std::ofstream file_;
    std::vector<std::string> names_;
    std::vector<Reader> readers_;
    bool header_written_ {false};
};

/*
 * --------------------------------------------------------------------------
 * CsvDatTrace
 * --------------------------------------------------------------------------
 */

class CsvDatTrace {
public:
    using Reader = std::function<std::string()>;

    CsvDatTrace() = default;

    CsvDatTrace(const std::string& csv_path, const std::string& dat_dir)
    {
        open(csv_path, dat_dir);
    }

    void open(const std::string& csv_path, const std::string& dat_dir)
    {
        csv_.open(csv_path);
        dat_dir_ = dat_dir;
        std::filesystem::create_directories(dat_dir_);
    }

    /*
     * Same representation for CSV and DAT.
     */
    void add_value(const std::string& name, Reader reader)
    {
        add_value(name, reader, reader);
    }

    /*
     * Different representations for CSV and DAT.
     *
     * Typical use:
     *   CSV -> human-readable decimal
     *   DAT -> raw signed decimal integer
     */
    void add_value(const std::string& name, Reader csv_reader, Reader dat_reader)
    {
        csv_.add_value(name, csv_reader);

        SignalDat dat;
        dat.name = name;
        dat.reader = dat_reader;

        const std::filesystem::path dat_path =
            std::filesystem::path(dat_dir_) / (sanitize_filename(name) + ".dat");

        const std::filesystem::path parent = dat_path.parent_path();

        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }

        dat.file.open(dat_path);

        if (!dat.file) {
            throw std::runtime_error("could not open DAT trace file: " + dat_path.string());
        }

        dat_files_.push_back(std::move(dat));
    }

    template <typename Getter>
    void add_signal(const std::string& name, Getter getter)
    {
        add_value(
            name,
            [getter]() {
                return value_to_string(getter());
            },
            [getter]() {
                return value_to_dat(getter());
            }
        );
    }

    void sample(std::uint64_t cycle)
    {
        csv_.sample(cycle);

        for (auto& dat : dat_files_) {
            dat.file << dat.reader() << "\n";
        }
    }

private:
    struct SignalDat {
        std::string name;
        Reader reader;
        std::ofstream file;
    };

    static std::string sanitize_filename(const std::string& name)
    {
        std::string out;
        out.reserve(name.size());

        for (unsigned char c : name) {
            if (std::isalnum(c) || c == '_' || c == '-' || c == '.') {
                out.push_back(static_cast<char>(c));
            }
            else {
                out.push_back('_');
            }
        }

        if (out.empty()) {
            return "signal";
        }

        return out;
    }

    CsvTrace csv_;
    std::string dat_dir_;
    std::vector<SignalDat> dat_files_;
};

/*
 * --------------------------------------------------------------------------
 * VectorTraceFiles
 * --------------------------------------------------------------------------
 *
 * Standard vector layout:
 *
 *   build/<CASE>/
 *     simulation/
 *       vectors/
 *         stimuli/
 *           in_ports.csv
 *           in_ports/
 *             <input_signal>.dat
 *         expected/
 *           out_ports.csv
 *           out_ports/
 *             <output_signal>.dat
 */

class VectorTraceFiles {
public:
    explicit VectorTraceFiles(const std::filesystem::path& case_dir)
        : case_dir_(case_dir),
          sim_dir_(case_dir_ / "simulation"),
          vectors_dir_(sim_dir_ / "vectors"),
          stimuli_dir_(vectors_dir_ / "stimuli"),
          expected_dir_(vectors_dir_ / "expected"),
          in_dat_dir_(stimuli_dir_ / "in_ports"),
          out_dat_dir_(expected_dir_ / "out_ports")
    {
        std::filesystem::create_directories(stimuli_dir_);
        std::filesystem::create_directories(expected_dir_);
        std::filesystem::create_directories(in_dat_dir_);
        std::filesystem::create_directories(out_dat_dir_);

        in_ports.open(
            in_ports_csv().string(),
            in_dat_dir_.string()
        );

        out_ports.open(
            out_ports_csv().string(),
            out_dat_dir_.string()
        );
    }

    template <typename T>
    void add_in_port(const std::string& name, const InPort<T>& port)
    {
        in_ports.add_signal(name, [&port]() {
            return port.value();
        });
    }

    template <typename T>
    void add_out_port(const std::string& name, const OutPort<T>& port)
    {
        out_ports.add_signal(name, [&port]() {
            return port.value();
        });
    }

    template <typename T>
    void add_in_wire(const std::string& name, const Wire<T>& wire)
    {
        in_ports.add_signal(name, [&wire]() {
            return wire.value();
        });
    }

    template <typename T>
    void add_out_wire(const std::string& name, const Wire<T>& wire)
    {
        out_ports.add_signal(name, [&wire]() {
            return wire.value();
        });
    }

    template <typename T>
    void add_in_reg_i(const std::string& name, const Reg<T>& reg)
    {
        in_ports.add_signal(name + ".i", [&reg]() {
            return reg.i;
        });
    }

    template <typename T>
    void add_in_reg_o(const std::string& name, const Reg<T>& reg)
    {
        in_ports.add_signal(name + ".o", [&reg]() {
            return reg.o;
        });
    }

    template <typename T>
    void add_out_reg_i(const std::string& name, const Reg<T>& reg)
    {
        out_ports.add_signal(name + ".i", [&reg]() {
            return reg.i;
        });
    }

    template <typename T>
    void add_reg_o(const std::string& name, const Reg<T>& reg)
    {
        out_ports.add_signal(name + ".o", [&reg]() {
            return reg.o;
        });
    }

    void sample(std::uint64_t cycle)
    {
        in_ports.sample(cycle);
        out_ports.sample(cycle);
    }

    void sample(const Simulator& sim)
    {
        sample(sim.cycle_count());
    }

    std::filesystem::path in_ports_csv() const
    {
        return stimuli_dir_ / "in_ports.csv";
    }

    std::filesystem::path out_ports_csv() const
    {
        return expected_dir_ / "out_ports.csv";
    }

    std::filesystem::path in_dat_dir() const
    {
        return in_dat_dir_;
    }

    std::filesystem::path out_dat_dir() const
    {
        return out_dat_dir_;
    }

    CsvDatTrace in_ports;
    CsvDatTrace out_ports;

private:
    std::filesystem::path case_dir_;
    std::filesystem::path sim_dir_;
    std::filesystem::path vectors_dir_;
    std::filesystem::path stimuli_dir_;
    std::filesystem::path expected_dir_;
    std::filesystem::path in_dat_dir_;
    std::filesystem::path out_dat_dir_;
};

/*
 * --------------------------------------------------------------------------
 * TraceFiles
 * --------------------------------------------------------------------------
 *
 * Legacy/general-purpose traces:
 *
 *   in_ports.csv
 *   out_ports.csv
 *   wires.csv
 *   regs.csv
 */

class TraceFiles {
public:
    explicit TraceFiles(const std::string& dir = "")
        : in_ports(make_path(dir, "in_ports.csv")),
          out_ports(make_path(dir, "out_ports.csv")),
          wires(make_path(dir, "wires.csv")),
          regs(make_path(dir, "regs.csv"))
    {
    }

    void sample(const Simulator& sim)
    {
        sample(sim.cycle_count());
    }

    void sample(std::uint64_t cycle)
    {
        in_ports.sample(cycle);
        out_ports.sample(cycle);
        wires.sample(cycle);
        regs.sample(cycle);
    }

    template <typename T>
    void add_in_port(const std::string& name, const InPort<T>& port)
    {
        in_ports.add_value(name, [&port]() {
            return value_to_string(port.value());
        });
    }

    template <typename T>
    void add_out_port(const std::string& name, const OutPort<T>& port)
    {
        out_ports.add_value(name, [&port]() {
            return value_to_string(port.value());
        });
    }

    template <typename T>
    void add_wire(const std::string& name, const Wire<T>& wire)
    {
        wires.add_value(name, [&wire]() {
            return value_to_string(wire.value());
        });
    }

    template <typename T>
    void add_reg(const std::string& name, const Reg<T>& reg)
    {
        regs.add_value(name, [&reg]() {
            return value_to_string(reg.o);
        });
    }

    template <typename T>
    void add_reg_i(const std::string& name, const Reg<T>& reg)
    {
        regs.add_value(name + ".i", [&reg]() {
            return value_to_string(reg.i);
        });
    }

    template <typename T>
    void add_reg_o(const std::string& name, const Reg<T>& reg)
    {
        regs.add_value(name + ".o", [&reg]() {
            return value_to_string(reg.o);
        });
    }

    CsvTrace in_ports;
    CsvTrace out_ports;
    CsvTrace wires;
    CsvTrace regs;

private:
    static std::string make_path(const std::string& dir, const std::string& file)
    {
        if (dir.empty()) {
            return file;
        }

        std::filesystem::create_directories(dir);

        return (std::filesystem::path(dir) / file).string();
    }
};

} // namespace rtl