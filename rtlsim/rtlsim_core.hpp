#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>
#include <ap_fixed.h>
#include <ostream>

namespace rtl {

class bit_t {
public:
    using storage_t = ap_ufixed<1, 1>;

    bit_t()
        : value_(0)
    {
    }

    bit_t(bool value)
        : value_(value ? 1 : 0)
    {
    }

    bit_t(int value)
        : value_(value != 0 ? 1 : 0)
    {
    }

    bit_t(const storage_t& value)
        : value_(value != 0 ? 1 : 0)
    {
    }

    bit_t& operator=(bool value)
    {
        value_ = value ? 1 : 0;
        return *this;
    }

    bit_t& operator=(int value)
    {
        value_ = value != 0 ? 1 : 0;
        return *this;
    }

    bit_t& operator=(const storage_t& value)
    {
        value_ = value != 0 ? 1 : 0;
        return *this;
    }

    bool to_bool() const
    {
        return value_ != 0;
    }

    storage_t raw() const
    {
        return value_;
    }

    explicit operator bool() const
    {
        return to_bool();
    }

private:
    storage_t value_;
};

inline bool operator==(const bit_t& a, const bit_t& b)
{
    return a.to_bool() == b.to_bool();
}

inline bool operator!=(const bit_t& a, const bit_t& b)
{
    return !(a == b);
}

inline std::ostream& operator<<(std::ostream& os, const bit_t& value)
{
    os << (value.to_bool() ? 1 : 0);
    return os;
}

class IReg {
public:
    virtual ~IReg() = default;

    virtual void hold() = 0;
    virtual void commit() = 0;
    virtual void reset() = 0;
};

template <typename T>
class Reg final : public IReg {
public:
    T i {};
    T o {};

    Reg() = default;

    explicit Reg(const T& reset_value)
        : i(reset_value),
          o(reset_value),
          reset_value_(reset_value),
          has_reset_value_(true)
    {
    }

    void set_reset_value(const T& value)
    {
        reset_value_ = value;
        has_reset_value_ = true;
    }

    void set_initial_value(const T& value)
    {
        i = value;
        o = value;
    }

    void reset() override
    {
        if (has_reset_value_) {
            i = reset_value_;
            o = reset_value_;
        }
        else {
            i = T {};
            o = T {};
        }
    }

    void hold() override
    {
        i = o;
    }

    void commit() override
    {
        o = i;
    }

private:
    T reset_value_ {};
    bool has_reset_value_ {false};
};

template <typename T>
class Signal {
public:
    Signal() = default;

    explicit Signal(const T& value)
        : value_(value)
    {
    }

    Signal& operator=(const T& value)
    {
        value_ = value;
        return *this;
    }

    operator const T&() const
    {
        return value_;
    }

    const T& value() const
    {
        return value_;
    }

    void set_initial_value(const T& value)
    {
        value_ = value;
    }

private:
    T value_ {};
};

template <typename T>
using Wire = Signal<T>;

enum class PortDirection {
    Input,
    Output,
};

template <typename T, PortDirection Direction>
class Port final : public Signal<T> {
public:
    using Signal<T>::Signal;
    using Signal<T>::operator=;

    static constexpr PortDirection direction = Direction;
};

template <typename T>
using InPort = Port<T, PortDirection::Input>;

template <typename T>
using OutPort = Port<T, PortDirection::Output>;

class ClockDomain {
public:
    void add(IReg& reg)
    {
        IReg* ptr = &reg;

        if (std::find(regs_.begin(), regs_.end(), ptr) == regs_.end()) {
            regs_.push_back(ptr);
        }
    }

    void reset_all()
    {
        for (IReg* reg : regs_) {
            reg->reset();
        }
    }

    void hold_all()
    {
        for (IReg* reg : regs_) {
            reg->hold();
        }
    }

    void commit_all()
    {
        for (IReg* reg : regs_) {
            reg->commit();
        }
    }

private:
    std::vector<IReg*> regs_;
};

class Module {
public:
    virtual ~Module() = default;

    virtual void connect_clocks(ClockDomain& clk)
    {
        (void)clk;
    }

    virtual void init()
    {
    }

    virtual void combinational()
    {
    }

    virtual void sequential()
    {
    }
};

class Simulator {
public:
    explicit Simulator(ClockDomain& clk)
        : clk_(clk)
    {
    }

    void add(Module& module)
    {
        Module* ptr = &module;

        if (std::find(modules_.begin(), modules_.end(), ptr) != modules_.end()) {
            return;
        }

        module.connect_clocks(clk_);
        modules_.push_back(ptr);
    }

    void init()
    {
        for (Module* module : modules_) {
            module->init();
        }

        clk_.hold_all();
        eval_combinational();

        cycle_ = 0;
    }

    void reset()
    {
        clk_.reset_all();
        eval_combinational();
        cycle_ = 0;
    }

    void eval_combinational()
    {
        for (Module* module : modules_) {
            module->combinational();
        }
    }

    void eval_sequential()
    {
        for (Module* module : modules_) {
            module->sequential();
        }
    }

    void cycle()
    {
        clk_.hold_all();
        eval_combinational();
        eval_sequential();
        clk_.commit_all();
        eval_combinational();
        cycle_++;
    }

    void cycle(std::uint64_t n)
    {
        for (std::uint64_t k = 0; k < n; ++k) {
            cycle();
        }
    }

    std::uint64_t cycle_count() const
    {
        return cycle_;
    }

private:
    ClockDomain& clk_;
    std::vector<Module*> modules_;
    std::uint64_t cycle_ {0};
};

} // namespace rtl