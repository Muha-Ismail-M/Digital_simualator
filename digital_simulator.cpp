#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <unordered_map>
#include <set>
#include <string>
#include <memory>
#include <functional>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <optional>
#include <variant>
#include <any>
#include <cassert>

namespace DigitalSimulator {

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

class Entity;
class Customer;
class Staff;
class Station;
class Queue;
class Event;
class SimulationEngine;
class BusinessModel;
class StatisticsCollector;

// ============================================================================
// UTILITY CLASSES
// ============================================================================

// High-precision time representation (in minutes from simulation start)
using SimTime = double;
constexpr SimTime INFINITY_TIME = std::numeric_limits<double>::max();

// Unique ID generator
class IDGenerator {
    static inline std::atomic<uint64_t> counter{0};
public:
    static uint64_t next() { return ++counter; }
    static void reset() { counter = 0; }
};

// JSON-like configuration structure
class Config {
public:
    using Value = std::variant<int, double, std::string, bool, 
                               std::vector<Config>, std::map<std::string, Config>>;
private:
    Value value_;
    
public:
    Config() : value_(0) {}
    Config(int v) : value_(v) {}
    Config(double v) : value_(v) {}
    Config(const std::string& v) : value_(v) {}
    Config(const char* v) : value_(std::string(v)) {}
    Config(bool v) : value_(v) {}
    Config(std::vector<Config> v) : value_(std::move(v)) {}
    Config(std::map<std::string, Config> v) : value_(std::move(v)) {}
    
    template<typename T>
    T get() const { return std::get<T>(value_); }
    
    template<typename T>
    T get(const T& default_val) const {
        try { return std::get<T>(value_); }
        catch (...) { return default_val; }
    }
    
    Config& operator[](const std::string& key) {
        if (!std::holds_alternative<std::map<std::string, Config>>(value_)) {
            value_ = std::map<std::string, Config>{};
        }
        return std::get<std::map<std::string, Config>>(value_)[key];
    }
    
    const Config& operator[](const std::string& key) const {
        static Config empty;
        if (auto* m = std::get_if<std::map<std::string, Config>>(&value_)) {
            auto it = m->find(key);
            if (it != m->end()) return it->second;
        }
        return empty;
    }
    
    bool contains(const std::string& key) const {
        if (auto* m = std::get_if<std::map<std::string, Config>>(&value_)) {
            return m->find(key) != m->end();
        }
        return false;
    }
};

// ============================================================================
// PROBABILITY DISTRIBUTIONS
// ============================================================================

class RandomGenerator {
    std::mt19937_64 engine_;
    
public:
    RandomGenerator() : engine_(std::random_device{}()) {}
    explicit RandomGenerator(uint64_t seed) : engine_(seed) {}
    
    void seed(uint64_t s) { engine_.seed(s); }
    
    // Uniform distribution [min, max]
    double uniform(double min, double max) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(engine_);
    }
    
    // Exponential distribution (for inter-arrival times)
    double exponential(double rate) {
        if (rate <= 0) return INFINITY_TIME;
        std::exponential_distribution<double> dist(rate);
        return dist(engine_);
    }
    
    // Normal/Gaussian distribution
    double normal(double mean, double stddev) {
        std::normal_distribution<double> dist(mean, stddev);
        return std::max(0.0, dist(engine_)); // Ensure non-negative
    }
    
    // Poisson distribution (for arrival counts)
    int poisson(double lambda) {
        std::poisson_distribution<int> dist(lambda);
        return dist(engine_);
    }
    
    // Triangular distribution (min, mode, max)
    double triangular(double min, double mode, double max) {
        double u = uniform(0, 1);
        double fc = (mode - min) / (max - min);
        if (u < fc) {
            return min + std::sqrt(u * (max - min) * (mode - min));
        }
        return max - std::sqrt((1 - u) * (max - min) * (max - mode));
    }
    
    // Log-normal distribution
    double lognormal(double mean, double stddev) {
        std::lognormal_distribution<double> dist(mean, stddev);
        return dist(engine_);
    }
    
    // Erlang distribution (shape k, rate lambda)
    double erlang(int k, double lambda) {
        double sum = 0;
        for (int i = 0; i < k; ++i) {
            sum += exponential(lambda);
        }
        return sum;
    }
    
    // Weighted random selection
    template<typename T>
    const T& weightedChoice(const std::vector<std::pair<T, double>>& choices) {
        double total = 0;
        for (const auto& [val, weight] : choices) total += weight;
        
        double r = uniform(0, total);
        double cumulative = 0;
        for (const auto& [val, weight] : choices) {
            cumulative += weight;
            if (r <= cumulative) return val;
        }
        return choices.back().first;
    }
};

// ============================================================================
// STATISTICAL COLLECTORS
// ============================================================================

class TimeSeriesStats {
    std::vector<std::pair<SimTime, double>> samples_;
    double sum_ = 0;
    double sum_sq_ = 0;
    double min_ = std::numeric_limits<double>::max();
    double max_ = std::numeric_limits<double>::lowest();
    
public:
    void record(SimTime time, double value) {
        samples_.emplace_back(time, value);
        sum_ += value;
        sum_sq_ += value * value;
        min_ = std::min(min_, value);
        max_ = std::max(max_, value);
    }
    
    size_t count() const { return samples_.size(); }
    double sum() const { return sum_; }
    double mean() const { return samples_.empty() ? 0 : sum_ / samples_.size(); }
    double min() const { return samples_.empty() ? 0 : min_; }
    double max() const { return samples_.empty() ? 0 : max_; }
    
    double variance() const {
        if (samples_.size() < 2) return 0;
        double n = samples_.size();
        return (sum_sq_ - (sum_ * sum_) / n) / (n - 1);
    }
    
    double stddev() const { return std::sqrt(variance()); }
    
    double percentile(double p) const {
        if (samples_.empty()) return 0;
        
        std::vector<double> values;
        values.reserve(samples_.size());
        for (const auto& [t, v] : samples_) values.push_back(v);
        std::sort(values.begin(), values.end());
        
        size_t idx = static_cast<size_t>(p * (values.size() - 1));
        return values[idx];
    }
    
    // 95% confidence interval
    std::pair<double, double> confidenceInterval95() const {
        if (samples_.size() < 2) return {mean(), mean()};
        double se = stddev() / std::sqrt(samples_.size());
        return {mean() - 1.96 * se, mean() + 1.96 * se};
    }
    
    const std::vector<std::pair<SimTime, double>>& samples() const { return samples_; }
};

class TimeWeightedStats {
    double weighted_sum_ = 0;
    double total_time_ = 0;
    double last_value_ = 0;
    SimTime last_time_ = 0;
    double min_ = std::numeric_limits<double>::max();
    double max_ = std::numeric_limits<double>::lowest();
    
public:
    void update(SimTime time, double value) {
        if (time > last_time_) {
            double duration = time - last_time_;
            weighted_sum_ += last_value_ * duration;
            total_time_ += duration;
        }
        min_ = std::min(min_, value);
        max_ = std::max(max_, value);
        last_value_ = value;
        last_time_ = time;
    }
    
    double timeWeightedAverage() const {
        return total_time_ > 0 ? weighted_sum_ / total_time_ : 0;
    }
    
    double min() const { return min_ == std::numeric_limits<double>::max() ? 0 : min_; }
    double max() const { return max_ == std::numeric_limits<double>::lowest() ? 0 : max_; }
};

// ============================================================================
// CORE ENTITIES
// ============================================================================

enum class EntityState {
    IDLE,
    WAITING,
    IN_SERVICE,
    COMPLETED,
    ABANDONED
};

class Entity {
protected:
    uint64_t id_;
    std::string name_;
    SimTime creation_time_;
    EntityState state_;
    std::map<std::string, std::any> attributes_;
    
public:
    Entity(const std::string& name = "")
        : id_(IDGenerator::next())
        , name_(name)
        , creation_time_(0)
        , state_(EntityState::IDLE) {}
    
    virtual ~Entity() = default;
    
    uint64_t id() const { return id_; }
    const std::string& name() const { return name_; }
    SimTime creationTime() const { return creation_time_; }
    EntityState state() const { return state_; }
    
    void setCreationTime(SimTime t) { creation_time_ = t; }
    void setState(EntityState s) { state_ = s; }
    
    template<typename T>
    void setAttribute(const std::string& key, T value) {
        attributes_[key] = value;
    }
    
    template<typename T>
    T getAttribute(const std::string& key, T default_value = T{}) const {
        auto it = attributes_.find(key);
        if (it != attributes_.end()) {
            try { return std::any_cast<T>(it->second); }
            catch (...) { return default_value; }
        }
        return default_value;
    }
};

// Customer entity
class Customer : public Entity {
public:
    enum class Type { REGULAR, PRIORITY, VIP };
    enum class Behavior { PATIENT, IMPATIENT };
    
private:
    Type type_;
    Behavior behavior_;
    SimTime arrival_time_ = 0;
    SimTime service_start_time_ = 0;
    SimTime service_end_time_ = 0;
    SimTime max_wait_time_;
    double revenue_value_;
    std::vector<uint64_t> visited_stations_;
    
public:
    Customer(Type type = Type::REGULAR, Behavior behavior = Behavior::PATIENT)
        : Entity("Customer")
        , type_(type)
        , behavior_(behavior)
        , max_wait_time_(INFINITY_TIME)
        , revenue_value_(0) {}
    
    Type type() const { return type_; }
    Behavior behavior() const { return behavior_; }
    SimTime arrivalTime() const { return arrival_time_; }
    SimTime serviceStartTime() const { return service_start_time_; }
    SimTime serviceEndTime() const { return service_end_time_; }
    SimTime maxWaitTime() const { return max_wait_time_; }
    double revenueValue() const { return revenue_value_; }
    
    void setArrivalTime(SimTime t) { arrival_time_ = t; }
    void setServiceStartTime(SimTime t) { service_start_time_ = t; }
    void setServiceEndTime(SimTime t) { service_end_time_ = t; }
    void setMaxWaitTime(SimTime t) { max_wait_time_ = t; }
    void setRevenueValue(double v) { revenue_value_ = v; }
    
    SimTime waitTime() const {
        if (service_start_time_ > 0) return service_start_time_ - arrival_time_;
        return 0;
    }
    
    SimTime serviceTime() const {
        if (service_end_time_ > 0) return service_end_time_ - service_start_time_;
        return 0;
    }
    
    SimTime totalTime() const {
        if (service_end_time_ > 0) return service_end_time_ - arrival_time_;
        return 0;
    }
    
    void addVisitedStation(uint64_t station_id) {
        visited_stations_.push_back(station_id);
    }
    
    const std::vector<uint64_t>& visitedStations() const { return visited_stations_; }
    
    int priority() const {
        switch (type_) {
            case Type::VIP: return 0;
            case Type::PRIORITY: return 1;
            default: return 2;
        }
    }
};

// Staff/Worker entity
class Staff : public Entity {
public:
    enum class Skill { JUNIOR, REGULAR, SENIOR, EXPERT };
    
private:
    Skill skill_;
    double efficiency_;  // Multiplier for service time
    SimTime shift_start_;
    SimTime shift_end_;
    SimTime total_busy_time_ = 0;
    SimTime last_busy_start_ = 0;
    int customers_served_ = 0;
    bool is_busy_ = false;
    uint64_t current_station_ = 0;
    
public:
    Staff(const std::string& name, Skill skill = Skill::REGULAR)
        : Entity(name)
        , skill_(skill)
        , efficiency_(skillToEfficiency(skill))
        , shift_start_(0)
        , shift_end_(INFINITY_TIME) {}
    
    Skill skill() const { return skill_; }
    double efficiency() const { return efficiency_; }
    bool isBusy() const { return is_busy_; }
    bool isAvailable(SimTime time) const {
        return !is_busy_ && time >= shift_start_ && time < shift_end_;
    }
    int customersServed() const { return customers_served_; }
    SimTime totalBusyTime() const { return total_busy_time_; }
    uint64_t currentStation() const { return current_station_; }
    
    void setShift(SimTime start, SimTime end) {
        shift_start_ = start;
        shift_end_ = end;
    }
    
    void startService(SimTime time, uint64_t station_id) {
        is_busy_ = true;
        last_busy_start_ = time;
        current_station_ = station_id;
    }
    
    void endService(SimTime time) {
        if (is_busy_) {
            total_busy_time_ += time - last_busy_start_;
            is_busy_ = false;
            customers_served_++;
            current_station_ = 0;
        }
    }
    
    double utilization(SimTime total_time) const {
        if (total_time <= 0) return 0;
        return total_busy_time_ / total_time;
    }
    
private:
    static double skillToEfficiency(Skill s) {
        switch (s) {
            case Skill::JUNIOR: return 0.7;
            case Skill::REGULAR: return 1.0;
            case Skill::SENIOR: return 1.2;
            case Skill::EXPERT: return 1.5;
            default: return 1.0;
        }
    }
};

// ============================================================================
// QUEUE SYSTEM
// ============================================================================

enum class QueueDiscipline {
    FIFO,           // First In First Out
    LIFO,           // Last In First Out
    PRIORITY,       // Based on customer priority
    SJF,            // Shortest Job First (if known)
    RANDOM          // Random selection
};

class Queue {
    uint64_t id_;
    std::string name_;
    QueueDiscipline discipline_;
    size_t max_capacity_;
    
    std::deque<std::shared_ptr<Customer>> customers_;
    TimeWeightedStats length_stats_;
    TimeSeriesStats wait_stats_;
    int total_arrivals_ = 0;
    int total_abandonments_ = 0;
    
public:
    Queue(const std::string& name, 
          QueueDiscipline discipline = QueueDiscipline::FIFO,
          size_t max_capacity = std::numeric_limits<size_t>::max())
        : id_(IDGenerator::next())
        , name_(name)
        , discipline_(discipline)
        , max_capacity_(max_capacity) {}
    
    uint64_t id() const { return id_; }
    const std::string& name() const { return name_; }
    size_t size() const { return customers_.size(); }
    bool empty() const { return customers_.empty(); }
    bool isFull() const { return customers_.size() >= max_capacity_; }
    size_t maxCapacity() const { return max_capacity_; }
    
    bool enqueue(std::shared_ptr<Customer> customer, SimTime time) {
        if (isFull()) return false;
        
        customer->setState(EntityState::WAITING);
        customer->setArrivalTime(time);
        
        if (discipline_ == QueueDiscipline::PRIORITY) {
            // Insert in priority order
            auto it = std::find_if(customers_.begin(), customers_.end(),
                [&](const auto& c) { return c->priority() > customer->priority(); });
            customers_.insert(it, customer);
        } else {
            customers_.push_back(customer);
        }
        
        total_arrivals_++;
        length_stats_.update(time, customers_.size());
        return true;
    }
    
    std::shared_ptr<Customer> dequeue(SimTime time) {
        if (empty()) return nullptr;
        
        std::shared_ptr<Customer> customer;
        
        switch (discipline_) {
            case QueueDiscipline::LIFO:
                customer = customers_.back();
                customers_.pop_back();
                break;
            case QueueDiscipline::RANDOM: {
                size_t idx = rand() % customers_.size();
                customer = customers_[idx];
                customers_.erase(customers_.begin() + idx);
                break;
            }
            default: // FIFO, PRIORITY (already sorted)
                customer = customers_.front();
                customers_.pop_front();
                break;
        }
        
        customer->setServiceStartTime(time);
        wait_stats_.record(time, customer->waitTime());
        length_stats_.update(time, customers_.size());
        
        return customer;
    }
    
    // Remove customers who exceeded their patience
    std::vector<std::shared_ptr<Customer>> processAbandonments(SimTime time) {
        std::vector<std::shared_ptr<Customer>> abandoned;
        
        auto it = customers_.begin();
        while (it != customers_.end()) {
            double waited = time - (*it)->arrivalTime();
            if (waited > (*it)->maxWaitTime()) {
                (*it)->setState(EntityState::ABANDONED);
                abandoned.push_back(*it);
                it = customers_.erase(it);
                total_abandonments_++;
            } else {
                ++it;
            }
        }
        
        if (!abandoned.empty()) {
            length_stats_.update(time, customers_.size());
        }
        
        return abandoned;
    }
    
    // Statistics
    double averageLength() const { return length_stats_.timeWeightedAverage(); }
    double maxLength() const { return length_stats_.max(); }
    double averageWait() const { return wait_stats_.mean(); }
    double maxWait() const { return wait_stats_.max(); }
    double p95Wait() const { return wait_stats_.percentile(0.95); }
    int totalArrivals() const { return total_arrivals_; }
    int totalAbandonments() const { return total_abandonments_; }
    double abandonmentRate() const {
        return total_arrivals_ > 0 ? 
            static_cast<double>(total_abandonments_) / total_arrivals_ : 0;
    }
};

// ============================================================================
// SERVICE STATION
// ============================================================================

class Station {
public:
    enum class Type {
        SINGLE_SERVER,      // One customer at a time
        MULTI_SERVER,       // Multiple parallel servers
        BATCH,              // Process multiple customers together
        SELF_SERVICE        // No staff needed
    };
    
private:
    uint64_t id_;
    std::string name_;
    Type type_;
    
    // Service time distribution parameters
    double base_service_time_;
    double service_time_variance_;
    
    // Capacity
    int num_servers_;
    int active_servers_ = 0;
    
    // Associated queue and staff
    std::shared_ptr<Queue> input_queue_;
    std::vector<std::shared_ptr<Staff>> assigned_staff_;
    
    // Routing
    std::vector<std::pair<std::shared_ptr<Station>, double>> next_stations_;
    
    // Statistics
    TimeWeightedStats utilization_stats_;
    TimeSeriesStats service_time_stats_;
    int total_served_ = 0;
    SimTime total_busy_time_ = 0;
    
public:
    Station(const std::string& name, 
            Type type = Type::SINGLE_SERVER,
            double base_service_time = 5.0,
            int num_servers = 1)
        : id_(IDGenerator::next())
        , name_(name)
        , type_(type)
        , base_service_time_(base_service_time)
        , service_time_variance_(base_service_time * 0.2) // 20% variance
        , num_servers_(num_servers)
        , input_queue_(std::make_shared<Queue>(name + "_queue")) {}
    
    uint64_t id() const { return id_; }
    const std::string& name() const { return name_; }
    Type type() const { return type_; }
    int numServers() const { return num_servers_; }
    int activeServers() const { return active_servers_; }
    int availableServers() const { return num_servers_ - active_servers_; }
    bool hasCapacity() const { return active_servers_ < num_servers_; }
    std::shared_ptr<Queue> inputQueue() const { return input_queue_; }
    
    void setServiceTimeVariance(double var) { service_time_variance_ = var; }
    void setNumServers(int n) { num_servers_ = n; }
    void setInputQueue(std::shared_ptr<Queue> q) { input_queue_ = q; }
    
    void assignStaff(std::shared_ptr<Staff> staff) {
        assigned_staff_.push_back(staff);
    }
    
    void addNextStation(std::shared_ptr<Station> station, double probability = 1.0) {
        next_stations_.emplace_back(station, probability);
    }
    
    // Get available staff member
    std::shared_ptr<Staff> getAvailableStaff(SimTime time) {
        for (auto& staff : assigned_staff_) {
            if (staff->isAvailable(time)) {
                return staff;
            }
        }
        return nullptr;
    }
    
    // Calculate actual service time considering staff efficiency
    double calculateServiceTime(RandomGenerator& rng, std::shared_ptr<Staff> staff = nullptr) {
        double service_time = rng.normal(base_service_time_, service_time_variance_);
        
        if (staff) {
            service_time /= staff->efficiency();
        }
        
        return std::max(0.1, service_time); // Minimum service time
    }
    
    void startService(SimTime time) {
        active_servers_++;
        utilization_stats_.update(time, static_cast<double>(active_servers_) / num_servers_);
    }
    
    void endService(SimTime time, double service_time) {
        active_servers_--;
        total_served_++;
        total_busy_time_ += service_time;
        service_time_stats_.record(time, service_time);
        utilization_stats_.update(time, static_cast<double>(active_servers_) / num_servers_);
    }
    
    // Select next station based on routing probabilities
    std::shared_ptr<Station> selectNextStation(RandomGenerator& rng) {
        if (next_stations_.empty()) return nullptr;
        
        double r = rng.uniform(0, 1);
        double cumulative = 0;
        
        for (const auto& [station, prob] : next_stations_) {
            cumulative += prob;
            if (r <= cumulative) return station;
        }
        
        return next_stations_.back().first;
    }
    
    // Statistics
    double utilization() const { return utilization_stats_.timeWeightedAverage(); }
    double averageServiceTime() const { return service_time_stats_.mean(); }
    int totalServed() const { return total_served_; }
    double throughputPerHour(SimTime total_time) const {
        return total_time > 0 ? (total_served_ * 60.0) / total_time : 0;
    }
};

// ============================================================================
// EVENT SYSTEM
// ============================================================================

enum class EventType {
    CUSTOMER_ARRIVAL,
    SERVICE_START,
    SERVICE_END,
    STAFF_SHIFT_START,
    STAFF_SHIFT_END,
    ABANDONMENT_CHECK,
    SIMULATION_END,
    CUSTOM
};

class Event {
    uint64_t id_;
    EventType type_;
    SimTime time_;
    int priority_;  // Lower = higher priority
    std::function<void()> action_;
    std::map<std::string, std::any> data_;
    
public:
    Event(EventType type, SimTime time, std::function<void()> action, int priority = 5)
        : id_(IDGenerator::next())
        , type_(type)
        , time_(time)
        , priority_(priority)
        , action_(std::move(action)) {}
    
    uint64_t id() const { return id_; }
    EventType type() const { return type_; }
    SimTime time() const { return time_; }
    int priority() const { return priority_; }
    
    void execute() { if (action_) action_(); }
    
    template<typename T>
    void setData(const std::string& key, T value) {
        data_[key] = value;
    }
    
    template<typename T>
    T getData(const std::string& key) const {
        auto it = data_.find(key);
        if (it != data_.end()) {
            return std::any_cast<T>(it->second);
        }
        return T{};
    }
    
    bool operator>(const Event& other) const {
        if (time_ != other.time_) return time_ > other.time_;
        return priority_ > other.priority_;
    }
};

// ============================================================================
// SIMULATION ENGINE
// ============================================================================

class SimulationEngine {
    // Time management
    SimTime current_time_ = 0;
    SimTime end_time_;
    
    // Event queue (min-heap)
    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> event_queue_;
    
    // Random number generation
    RandomGenerator rng_;
    
    // Entity containers
    std::vector<std::shared_ptr<Customer>> all_customers_;
    std::vector<std::shared_ptr<Staff>> all_staff_;
    std::vector<std::shared_ptr<Station>> all_stations_;
    std::vector<std::shared_ptr<Queue>> all_queues_;
    
    // Statistics
    int total_events_processed_ = 0;
    std::chrono::steady_clock::time_point real_start_time_;
    
    // State
    bool running_ = false;
    bool paused_ = false;
    
public:
    SimulationEngine(SimTime end_time = 480.0)  // Default: 8-hour simulation
        : end_time_(end_time) {}
    
    // Time accessors
    SimTime currentTime() const { return current_time_; }
    SimTime endTime() const { return end_time_; }
    void setEndTime(SimTime t) { end_time_ = t; }
    
    // Random generator access
    RandomGenerator& rng() { return rng_; }
    void setSeed(uint64_t seed) { rng_.seed(seed); }
    
    // Entity registration
    void registerStation(std::shared_ptr<Station> station) {
        all_stations_.push_back(station);
        all_queues_.push_back(station->inputQueue());
    }
    
    void registerStaff(std::shared_ptr<Staff> staff) {
        all_staff_.push_back(staff);
    }
    
    void registerCustomer(std::shared_ptr<Customer> customer) {
        all_customers_.push_back(customer);
    }
    
    // Event scheduling
    void scheduleEvent(Event event) {
        event_queue_.push(std::move(event));
    }
    
    void scheduleEvent(EventType type, SimTime time, std::function<void()> action, int priority = 5) {
        event_queue_.push(Event(type, time, std::move(action), priority));
    }
    
    // Run simulation
    void run() {
        real_start_time_ = std::chrono::steady_clock::now();
        running_ = true;
        
        // Schedule end event
        scheduleEvent(EventType::SIMULATION_END, end_time_, [this]() {
            running_ = false;
        }, 0);
        
        // Main simulation loop
        while (running_ && !event_queue_.empty()) {
            if (paused_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            
            Event event = event_queue_.top();
            event_queue_.pop();
            
            if (event.time() > end_time_) break;
            
            current_time_ = event.time();
            event.execute();
            total_events_processed_++;
        }
        
        running_ = false;
    }
    
    void pause() { paused_ = true; }
    void resume() { paused_ = false; }
    void stop() { running_ = false; }
    
    // Statistics
    int totalEventsProcessed() const { return total_events_processed_; }
    
    double realTimeSeconds() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - real_start_time_).count();
    }
    
    const std::vector<std::shared_ptr<Customer>>& customers() const { return all_customers_; }
    const std::vector<std::shared_ptr<Staff>>& staff() const { return all_staff_; }
    const std::vector<std::shared_ptr<Station>>& stations() const { return all_stations_; }
    const std::vector<std::shared_ptr<Queue>>& queues() const { return all_queues_; }
    
    void reset() {
        current_time_ = 0;
        total_events_processed_ = 0;
        while (!event_queue_.empty()) event_queue_.pop();
        all_customers_.clear();
        all_staff_.clear();
        all_stations_.clear();
        all_queues_.clear();
        IDGenerator::reset();
    }
};

// ============================================================================
// BUSINESS MODEL BASE CLASS
// ============================================================================

struct SimulationResults {
    // Overall metrics
    SimTime total_simulation_time = 0;
    int total_customers = 0;
    int served_customers = 0;
    int abandoned_customers = 0;
    double abandonment_rate = 0;
    
    // Wait time metrics
    double avg_wait_time = 0;
    double max_wait_time = 0;
    double p95_wait_time = 0;
    
    // Service metrics
    double avg_service_time = 0;
    double avg_total_time = 0;
    
    // Resource metrics
    double avg_staff_utilization = 0;
    double avg_station_utilization = 0;
    
    // Financial metrics
    double total_revenue = 0;
    double lost_revenue = 0;
    
    // Throughput
    double customers_per_hour = 0;
    
    // Bottleneck analysis
    std::string primary_bottleneck;
    double bottleneck_utilization = 0;
    
    // Queue metrics per station
    struct StationMetrics {
        std::string name;
        double utilization;
        double avg_queue_length;
        double avg_wait_time;
        int served;
    };
    std::vector<StationMetrics> station_metrics;
    
    // Staff metrics
    struct StaffMetrics {
        std::string name;
        double utilization;
        int customers_served;
    };
    std::vector<StaffMetrics> staff_metrics;
};

class BusinessModel {
protected:
    std::string name_;
    std::string type_;
    Config config_;
    std::shared_ptr<SimulationEngine> engine_;
    
public:
    BusinessModel(const std::string& name, const std::string& type)
        : name_(name)
        , type_(type)
        , engine_(std::make_shared<SimulationEngine>()) {}
    
    virtual ~BusinessModel() = default;
    
    const std::string& name() const { return name_; }
    const std::string& type() const { return type_; }
    std::shared_ptr<SimulationEngine> engine() { return engine_; }
    
    void configure(const Config& config) {
        config_ = config;
        setup();
    }
    
    virtual void setup() = 0;
    virtual void scheduleInitialEvents() = 0;
    
    void run(SimTime duration) {
        engine_->reset();
        engine_->setEndTime(duration);
        setup();
        scheduleInitialEvents();
        engine_->run();
    }
    
    SimulationResults collectResults() {
        SimulationResults results;
        results.total_simulation_time = engine_->currentTime();
        
        // Customer metrics
        auto& customers = engine_->customers();
        results.total_customers = customers.size();
        
        double total_wait = 0, max_wait = 0;
        double total_service = 0, total_time = 0;
        double total_revenue = 0, lost_revenue = 0;
        std::vector<double> wait_times;
        
        for (const auto& c : customers) {
            if (c->state() == EntityState::COMPLETED) {
                results.served_customers++;
                total_wait += c->waitTime();
                max_wait = std::max(max_wait, c->waitTime());
                wait_times.push_back(c->waitTime());
                total_service += c->serviceTime();
                total_time += c->totalTime();
                total_revenue += c->revenueValue();
            } else if (c->state() == EntityState::ABANDONED) {
                results.abandoned_customers++;
                lost_revenue += c->revenueValue();
            }
        }
        
        if (results.served_customers > 0) {
            results.avg_wait_time = total_wait / results.served_customers;
            results.avg_service_time = total_service / results.served_customers;
            results.avg_total_time = total_time / results.served_customers;
            
            // P95 wait time
            std::sort(wait_times.begin(), wait_times.end());
            size_t p95_idx = static_cast<size_t>(0.95 * wait_times.size());
            if (p95_idx < wait_times.size()) {
                results.p95_wait_time = wait_times[p95_idx];
            }
        }
        
        results.max_wait_time = max_wait;
        results.total_revenue = total_revenue;
        results.lost_revenue = lost_revenue;
        results.abandonment_rate = results.total_customers > 0 ?
            static_cast<double>(results.abandoned_customers) / results.total_customers : 0;
        results.customers_per_hour = results.total_simulation_time > 0 ?
            (results.served_customers * 60.0) / results.total_simulation_time : 0;
        
        // Staff metrics
        double total_staff_util = 0;
        for (const auto& s : engine_->staff()) {
            SimulationResults::StaffMetrics sm;
            sm.name = s->name();
            sm.utilization = s->utilization(results.total_simulation_time);
            sm.customers_served = s->customersServed();
            results.staff_metrics.push_back(sm);
            total_staff_util += sm.utilization;
        }
        results.avg_staff_utilization = engine_->staff().empty() ? 0 :
            total_staff_util / engine_->staff().size();
        
        // Station metrics and bottleneck detection
        double total_station_util = 0;
        for (const auto& station : engine_->stations()) {
            SimulationResults::StationMetrics sm;
            sm.name = station->name();
            sm.utilization = station->utilization();
            sm.avg_queue_length = station->inputQueue()->averageLength();
            sm.avg_wait_time = station->inputQueue()->averageWait();
            sm.served = station->totalServed();
            results.station_metrics.push_back(sm);
            total_station_util += sm.utilization;
            
            // Track bottleneck
            if (sm.utilization > results.bottleneck_utilization) {
                results.bottleneck_utilization = sm.utilization;
                results.primary_bottleneck = sm.name;
            }
        }
        results.avg_station_utilization = engine_->stations().empty() ? 0 :
            total_station_util / engine_->stations().size();
        
        return results;
    }
};

// ============================================================================
// RESTAURANT MODEL
// ============================================================================

class RestaurantModel : public BusinessModel {
    std::shared_ptr<Station> host_station_;
    std::shared_ptr<Station> seating_station_;
    std::shared_ptr<Station> ordering_station_;
    std::shared_ptr<Station> kitchen_station_;
    std::shared_ptr<Station> serving_station_;
    std::shared_ptr<Station> payment_station_;
    
    std::vector<std::shared_ptr<Staff>> hosts_;
    std::vector<std::shared_ptr<Staff>> servers_;
    std::vector<std::shared_ptr<Staff>> cooks_;
    std::vector<std::shared_ptr<Staff>> cashiers_;
    
    // Configuration
    int num_tables_ = 20;
    int num_hosts_ = 1;
    int num_servers_ = 3;
    int num_cooks_ = 2;
    int num_cashiers_ = 1;
    double arrival_rate_ = 30.0;  // customers per hour
    double avg_order_value_ = 25.0;
    double customer_patience_ = 15.0;  // minutes
    
public:
    RestaurantModel(const std::string& name = "Restaurant")
        : BusinessModel(name, "restaurant") {}
    
    void setup() override {
        // Read configuration
        if (config_.contains("num_tables")) num_tables_ = config_["num_tables"].get<int>();
        if (config_.contains("num_hosts")) num_hosts_ = config_["num_hosts"].get<int>();
        if (config_.contains("num_servers")) num_servers_ = config_["num_servers"].get<int>();
        if (config_.contains("num_cooks")) num_cooks_ = config_["num_cooks"].get<int>();
        if (config_.contains("num_cashiers")) num_cashiers_ = config_["num_cashiers"].get<int>();
        if (config_.contains("arrival_rate")) arrival_rate_ = config_["arrival_rate"].get<double>();
        if (config_.contains("avg_order_value")) avg_order_value_ = config_["avg_order_value"].get<double>();
        if (config_.contains("customer_patience")) customer_patience_ = config_["customer_patience"].get<double>();
        
        // Create stations
        host_station_ = std::make_shared<Station>("Host/Check-in", Station::Type::SINGLE_SERVER, 1.0, num_hosts_);
        seating_station_ = std::make_shared<Station>("Seating", Station::Type::MULTI_SERVER, 2.0, num_tables_);
        ordering_station_ = std::make_shared<Station>("Order Taking", Station::Type::SINGLE_SERVER, 4.0, num_servers_);
        kitchen_station_ = std::make_shared<Station>("Kitchen", Station::Type::MULTI_SERVER, 15.0, num_cooks_);
        serving_station_ = std::make_shared<Station>("Food Serving", Station::Type::SINGLE_SERVER, 2.0, num_servers_);
        payment_station_ = std::make_shared<Station>("Payment", Station::Type::SINGLE_SERVER, 3.0, num_cashiers_);
        
        // Set up flow
        host_station_->addNextStation(seating_station_);
        seating_station_->addNextStation(ordering_station_);
        ordering_station_->addNextStation(kitchen_station_);
        kitchen_station_->addNextStation(serving_station_);
        serving_station_->addNextStation(payment_station_);
        
        // Create staff
        for (int i = 0; i < num_hosts_; i++) {
            auto host = std::make_shared<Staff>("Host " + std::to_string(i+1), Staff::Skill::REGULAR);
            hosts_.push_back(host);
            host_station_->assignStaff(host);
            engine_->registerStaff(host);
        }
        
        for (int i = 0; i < num_servers_; i++) {
            auto server = std::make_shared<Staff>("Server " + std::to_string(i+1), Staff::Skill::REGULAR);
            servers_.push_back(server);
            ordering_station_->assignStaff(server);
            serving_station_->assignStaff(server);
            engine_->registerStaff(server);
        }
        
        for (int i = 0; i < num_cooks_; i++) {
            auto cook = std::make_shared<Staff>("Cook " + std::to_string(i+1), Staff::Skill::REGULAR);
            cooks_.push_back(cook);
            kitchen_station_->assignStaff(cook);
            engine_->registerStaff(cook);
        }
        
        for (int i = 0; i < num_cashiers_; i++) {
            auto cashier = std::make_shared<Staff>("Cashier " + std::to_string(i+1), Staff::Skill::REGULAR);
            cashiers_.push_back(cashier);
            payment_station_->assignStaff(cashier);
            engine_->registerStaff(cashier);
        }
        
        // Register stations
        engine_->registerStation(host_station_);
        engine_->registerStation(seating_station_);
        engine_->registerStation(ordering_station_);
        engine_->registerStation(kitchen_station_);
        engine_->registerStation(serving_station_);
        engine_->registerStation(payment_station_);
    }
    
    void scheduleInitialEvents() override {
        scheduleNextArrival(0);
    }
    
private:
    void scheduleNextArrival(SimTime after_time) {
        // Inter-arrival time based on arrival rate
        double rate_per_minute = arrival_rate_ / 60.0;
        SimTime inter_arrival = engine_->rng().exponential(rate_per_minute);
        SimTime arrival_time = after_time + inter_arrival;
        
        if (arrival_time < engine_->endTime()) {
            engine_->scheduleEvent(EventType::CUSTOMER_ARRIVAL, arrival_time, [this, arrival_time]() {
                handleCustomerArrival(arrival_time);
                scheduleNextArrival(arrival_time);
            });
        }
    }
    
    void handleCustomerArrival(SimTime time) {
        // Create customer
        auto customer = std::make_shared<Customer>(Customer::Type::REGULAR, Customer::Behavior::IMPATIENT);
        customer->setCreationTime(time);
        customer->setArrivalTime(time);
        customer->setMaxWaitTime(engine_->rng().normal(customer_patience_, 3.0));
        customer->setRevenueValue(engine_->rng().normal(avg_order_value_, avg_order_value_ * 0.3));
        engine_->registerCustomer(customer);
        
        // Try to enter first station
        if (host_station_->inputQueue()->enqueue(customer, time)) {
            tryStartService(host_station_, time);
        }
    }
    
    void tryStartService(std::shared_ptr<Station> station, SimTime time) {
        while (!station->inputQueue()->empty() && station->hasCapacity()) {
            auto staff = station->getAvailableStaff(time);
            if (!staff && station->type() != Station::Type::SELF_SERVICE) {
                break;  // No staff available
            }
            
            auto customer = station->inputQueue()->dequeue(time);
            if (!customer) break;
            
            customer->setState(EntityState::IN_SERVICE);
            station->startService(time);
            if (staff) staff->startService(time, station->id());
            
            // Calculate service time
            double service_time = station->calculateServiceTime(engine_->rng(), staff);
            SimTime end_time = time + service_time;
            
            // Schedule service end
            engine_->scheduleEvent(EventType::SERVICE_END, end_time, 
                [this, station, customer, staff, end_time, service_time]() {
                    handleServiceEnd(station, customer, staff, end_time, service_time);
                });
        }
    }
    
    void handleServiceEnd(std::shared_ptr<Station> station,
                          std::shared_ptr<Customer> customer,
                          std::shared_ptr<Staff> staff,
                          SimTime time,
                          double service_time) {
        station->endService(time, service_time);
        if (staff) staff->endService(time);
        customer->addVisitedStation(station->id());
        
        // Route to next station
        auto next_station = station->selectNextStation(engine_->rng());
        if (next_station) {
            if (next_station->inputQueue()->enqueue(customer, time)) {
                tryStartService(next_station, time);
            }
        } else {
            // Customer exits system
            customer->setServiceEndTime(time);
            customer->setState(EntityState::COMPLETED);
        }
        
        // Try to serve next customer at current station
        tryStartService(station, time);
    }
};

// ============================================================================
// RETAIL STORE MODEL
// ============================================================================

class RetailStoreModel : public BusinessModel {
    std::shared_ptr<Station> entrance_station_;
    std::shared_ptr<Station> browsing_station_;
    std::shared_ptr<Station> fitting_room_station_;
    std::shared_ptr<Station> checkout_station_;
    
    int num_checkouts_ = 3;
    int num_fitting_rooms_ = 4;
    double arrival_rate_ = 60.0;
    double avg_transaction_ = 45.0;
    double fitting_room_prob_ = 0.3;
    
public:
    RetailStoreModel(const std::string& name = "Retail Store")
        : BusinessModel(name, "retail") {}
    
    void setup() override {
        if (config_.contains("num_checkouts")) num_checkouts_ = config_["num_checkouts"].get<int>();
        if (config_.contains("num_fitting_rooms")) num_fitting_rooms_ = config_["num_fitting_rooms"].get<int>();
        if (config_.contains("arrival_rate")) arrival_rate_ = config_["arrival_rate"].get<double>();
        if (config_.contains("avg_transaction")) avg_transaction_ = config_["avg_transaction"].get<double>();
        if (config_.contains("fitting_room_prob")) fitting_room_prob_ = config_["fitting_room_prob"].get<double>();
        
        // Create stations
        entrance_station_ = std::make_shared<Station>("Entrance", Station::Type::SELF_SERVICE, 0.5, 100);
        browsing_station_ = std::make_shared<Station>("Shopping Floor", Station::Type::SELF_SERVICE, 15.0, 200);
        fitting_room_station_ = std::make_shared<Station>("Fitting Rooms", Station::Type::SELF_SERVICE, 8.0, num_fitting_rooms_);
        checkout_station_ = std::make_shared<Station>("Checkout", Station::Type::MULTI_SERVER, 3.0, num_checkouts_);
        
        // Create checkout staff
        for (int i = 0; i < num_checkouts_; i++) {
            auto cashier = std::make_shared<Staff>("Cashier " + std::to_string(i+1), Staff::Skill::REGULAR);
            checkout_station_->assignStaff(cashier);
            engine_->registerStaff(cashier);
        }
        
        // Set up flow with probabilities
        entrance_station_->addNextStation(browsing_station_, 1.0);
        browsing_station_->addNextStation(fitting_room_station_, fitting_room_prob_);
        browsing_station_->addNextStation(checkout_station_, 1.0 - fitting_room_prob_);
        fitting_room_station_->addNextStation(checkout_station_, 1.0);
        
        engine_->registerStation(entrance_station_);
        engine_->registerStation(browsing_station_);
        engine_->registerStation(fitting_room_station_);
        engine_->registerStation(checkout_station_);
    }
    
    void scheduleInitialEvents() override {
        scheduleNextArrival(0);
    }
    
private:
    void scheduleNextArrival(SimTime after_time) {
        double rate_per_minute = arrival_rate_ / 60.0;
        SimTime inter_arrival = engine_->rng().exponential(rate_per_minute);
        SimTime arrival_time = after_time + inter_arrival;
        
        if (arrival_time < engine_->endTime()) {
            engine_->scheduleEvent(EventType::CUSTOMER_ARRIVAL, arrival_time, [this, arrival_time]() {
                handleCustomerArrival(arrival_time);
                scheduleNextArrival(arrival_time);
            });
        }
    }
    
    void handleCustomerArrival(SimTime time) {
        auto customer = std::make_shared<Customer>();
        customer->setCreationTime(time);
        customer->setArrivalTime(time);
        customer->setRevenueValue(engine_->rng().lognormal(std::log(avg_transaction_), 0.5));
        customer->setMaxWaitTime(engine_->rng().normal(10.0, 3.0));
        engine_->registerCustomer(customer);
        
        // Direct processing for self-service stations
        processCustomerAtStation(customer, entrance_station_, time);
    }
    
    void processCustomerAtStation(std::shared_ptr<Customer> customer,
                                   std::shared_ptr<Station> station,
                                   SimTime time) {
        if (station->type() == Station::Type::SELF_SERVICE) {
            // Self-service: immediate processing with random duration
            double service_time = station->calculateServiceTime(engine_->rng());
            SimTime end_time = time + service_time;
            
            customer->setState(EntityState::IN_SERVICE);
            
            engine_->scheduleEvent(EventType::SERVICE_END, end_time,
                [this, customer, station, end_time, service_time]() {
                    station->endService(end_time, service_time);
                    customer->addVisitedStation(station->id());
                    
                    auto next = station->selectNextStation(engine_->rng());
                    if (next) {
                        processCustomerAtStation(customer, next, end_time);
                    } else {
                        customer->setServiceEndTime(end_time);
                        customer->setState(EntityState::COMPLETED);
                    }
                });
        } else {
            // Staffed station: queue and wait
            if (station->inputQueue()->enqueue(customer, time)) {
                tryStartService(station, time);
            }
        }
    }
    
    void tryStartService(std::shared_ptr<Station> station, SimTime time) {
        while (!station->inputQueue()->empty() && station->hasCapacity()) {
            auto staff = station->getAvailableStaff(time);
            if (!staff) break;
            
            auto customer = station->inputQueue()->dequeue(time);
            if (!customer) break;
            
            customer->setState(EntityState::IN_SERVICE);
            station->startService(time);
            staff->startService(time, station->id());
            
            double service_time = station->calculateServiceTime(engine_->rng(), staff);
            SimTime end_time = time + service_time;
            
            engine_->scheduleEvent(EventType::SERVICE_END, end_time,
                [this, station, customer, staff, end_time, service_time]() {
                    station->endService(end_time, service_time);
                    staff->endService(end_time);
                    customer->addVisitedStation(station->id());
                    
                    auto next = station->selectNextStation(engine_->rng());
                    if (next) {
                        processCustomerAtStation(customer, next, end_time);
                    } else {
                        customer->setServiceEndTime(end_time);
                        customer->setState(EntityState::COMPLETED);
                    }
                    
                    tryStartService(station, end_time);
                });
        }
    }
};

// ============================================================================
// WAREHOUSE MODEL
// ============================================================================

class WarehouseModel : public BusinessModel {
    std::shared_ptr<Station> receiving_station_;
    std::shared_ptr<Station> inspection_station_;
    std::shared_ptr<Station> storage_station_;
    std::shared_ptr<Station> picking_station_;
    std::shared_ptr<Station> packing_station_;
    std::shared_ptr<Station> shipping_station_;
    
    int num_dock_doors_ = 4;
    int num_inspectors_ = 2;
    int num_forklift_operators_ = 3;
    int num_pickers_ = 5;
    int num_packers_ = 3;
    double inbound_rate_ = 10.0;  // shipments per hour
    double order_rate_ = 20.0;    // orders per hour
    
public:
    WarehouseModel(const std::string& name = "Warehouse")
        : BusinessModel(name, "warehouse") {}
    
    void setup() override {
        if (config_.contains("num_dock_doors")) num_dock_doors_ = config_["num_dock_doors"].get<int>();
        if (config_.contains("num_inspectors")) num_inspectors_ = config_["num_inspectors"].get<int>();
        if (config_.contains("num_forklift_operators")) num_forklift_operators_ = config_["num_forklift_operators"].get<int>();
        if (config_.contains("num_pickers")) num_pickers_ = config_["num_pickers"].get<int>();
        if (config_.contains("num_packers")) num_packers_ = config_["num_packers"].get<int>();
        if (config_.contains("inbound_rate")) inbound_rate_ = config_["inbound_rate"].get<double>();
        if (config_.contains("order_rate")) order_rate_ = config_["order_rate"].get<double>();
        
        // Create stations for inbound flow
        receiving_station_ = std::make_shared<Station>("Receiving Dock", Station::Type::MULTI_SERVER, 20.0, num_dock_doors_);
        inspection_station_ = std::make_shared<Station>("QC Inspection", Station::Type::MULTI_SERVER, 10.0, num_inspectors_);
        storage_station_ = std::make_shared<Station>("Put-away", Station::Type::MULTI_SERVER, 15.0, num_forklift_operators_);
        
        // Create stations for outbound flow
        picking_station_ = std::make_shared<Station>("Order Picking", Station::Type::MULTI_SERVER, 12.0, num_pickers_);
        packing_station_ = std::make_shared<Station>("Packing", Station::Type::MULTI_SERVER, 8.0, num_packers_);
        shipping_station_ = std::make_shared<Station>("Shipping", Station::Type::MULTI_SERVER, 5.0, num_dock_doors_);
        
        // Create staff
        for (int i = 0; i < num_inspectors_; i++) {
            auto inspector = std::make_shared<Staff>("Inspector " + std::to_string(i+1), Staff::Skill::REGULAR);
            inspection_station_->assignStaff(inspector);
            engine_->registerStaff(inspector);
        }
        
        for (int i = 0; i < num_forklift_operators_; i++) {
            auto operator_ = std::make_shared<Staff>("Forklift Op " + std::to_string(i+1), Staff::Skill::REGULAR);
            receiving_station_->assignStaff(operator_);
            storage_station_->assignStaff(operator_);
            engine_->registerStaff(operator_);
        }
        
        for (int i = 0; i < num_pickers_; i++) {
            auto picker = std::make_shared<Staff>("Picker " + std::to_string(i+1), Staff::Skill::REGULAR);
            picking_station_->assignStaff(picker);
            engine_->registerStaff(picker);
        }
        
        for (int i = 0; i < num_packers_; i++) {
            auto packer = std::make_shared<Staff>("Packer " + std::to_string(i+1), Staff::Skill::REGULAR);
            packing_station_->assignStaff(packer);
            shipping_station_->assignStaff(packer);
            engine_->registerStaff(packer);
        }
        
        // Set up flows
        receiving_station_->addNextStation(inspection_station_);
        inspection_station_->addNextStation(storage_station_);
        
        picking_station_->addNextStation(packing_station_);
        packing_station_->addNextStation(shipping_station_);
        
        engine_->registerStation(receiving_station_);
        engine_->registerStation(inspection_station_);
        engine_->registerStation(storage_station_);
        engine_->registerStation(picking_station_);
        engine_->registerStation(packing_station_);
        engine_->registerStation(shipping_station_);
    }
    
    void scheduleInitialEvents() override {
        scheduleNextInbound(0);
        scheduleNextOrder(0);
    }
    
private:
    void scheduleNextInbound(SimTime after_time) {
        double rate_per_minute = inbound_rate_ / 60.0;
        SimTime inter_arrival = engine_->rng().exponential(rate_per_minute);
        SimTime arrival_time = after_time + inter_arrival;
        
        if (arrival_time < engine_->endTime()) {
            engine_->scheduleEvent(EventType::CUSTOMER_ARRIVAL, arrival_time, [this, arrival_time]() {
                handleInboundShipment(arrival_time);
                scheduleNextInbound(arrival_time);
            });
        }
    }
    
    void scheduleNextOrder(SimTime after_time) {
        double rate_per_minute = order_rate_ / 60.0;
        SimTime inter_arrival = engine_->rng().exponential(rate_per_minute);
        SimTime arrival_time = after_time + inter_arrival;
        
        if (arrival_time < engine_->endTime()) {
            engine_->scheduleEvent(EventType::CUSTOMER_ARRIVAL, arrival_time, [this, arrival_time]() {
                handleOrder(arrival_time);
                scheduleNextOrder(arrival_time);
            });
        }
    }
    
    void handleInboundShipment(SimTime time) {
        auto shipment = std::make_shared<Customer>();
        shipment->setCreationTime(time);
        shipment->setArrivalTime(time);
        shipment->setAttribute("type", std::string("inbound"));
        engine_->registerCustomer(shipment);
        
        receiving_station_->inputQueue()->enqueue(shipment, time);
        tryProcessInbound(time);
    }
    
    void handleOrder(SimTime time) {
        auto order = std::make_shared<Customer>();
        order->setCreationTime(time);
        order->setArrivalTime(time);
        order->setAttribute("type", std::string("order"));
        order->setRevenueValue(engine_->rng().normal(100.0, 30.0));
        engine_->registerCustomer(order);
        
        picking_station_->inputQueue()->enqueue(order, time);
        tryProcessOrder(time);
    }
    
    void tryProcessInbound(SimTime time) {
        processAtStation(receiving_station_, time, [this](auto c, auto t) { 
            inspection_station_->inputQueue()->enqueue(c, t);
            processAtStation(inspection_station_, t, [this](auto c2, auto t2) {
                storage_station_->inputQueue()->enqueue(c2, t2);
                processAtStation(storage_station_, t2, [](auto c3, auto t3) {
                    c3->setServiceEndTime(t3);
                    c3->setState(EntityState::COMPLETED);
                });
            });
        });
    }
    
    void tryProcessOrder(SimTime time) {
        processAtStation(picking_station_, time, [this](auto c, auto t) {
            packing_station_->inputQueue()->enqueue(c, t);
            processAtStation(packing_station_, t, [this](auto c2, auto t2) {
                shipping_station_->inputQueue()->enqueue(c2, t2);
                processAtStation(shipping_station_, t2, [](auto c3, auto t3) {
                    c3->setServiceEndTime(t3);
                    c3->setState(EntityState::COMPLETED);
                });
            });
        });
    }
    
    void processAtStation(std::shared_ptr<Station> station, SimTime time,
                          std::function<void(std::shared_ptr<Customer>, SimTime)> on_complete) {
        while (!station->inputQueue()->empty() && station->hasCapacity()) {
            auto staff = station->getAvailableStaff(time);
            if (!staff) break;
            
            auto item = station->inputQueue()->dequeue(time);
            if (!item) break;
            
            item->setState(EntityState::IN_SERVICE);
            station->startService(time);
            staff->startService(time, station->id());
            
            double service_time = station->calculateServiceTime(engine_->rng(), staff);
            SimTime end_time = time + service_time;
            
            engine_->scheduleEvent(EventType::SERVICE_END, end_time,
                [this, station, item, staff, end_time, service_time, on_complete]() {
                    station->endService(end_time, service_time);
                    staff->endService(end_time);
                    item->addVisitedStation(station->id());
                    
                    on_complete(item, end_time);
                    
                    // Process next item
                    if (station == picking_station_) tryProcessOrder(end_time);
                    else tryProcessInbound(end_time);
                });
        }
    }
};

// ============================================================================
// SCENARIO MANAGER
// ============================================================================

struct Scenario {
    std::string name;
    std::string description;
    Config config;
    SimulationResults results;
    bool completed = false;
};

class ScenarioManager {
    std::vector<Scenario> scenarios_;
    std::string base_model_type_;
    std::string business_name_;
    
public:
    ScenarioManager(const std::string& model_type, const std::string& name)
        : base_model_type_(model_type)
        , business_name_(name) {}
    
    void addScenario(const std::string& name, const std::string& desc, const Config& config) {
        scenarios_.push_back({name, desc, config, {}, false});
    }
    
    void runAll(SimTime duration, int replications = 1, uint64_t base_seed = 12345) {
        for (auto& scenario : scenarios_) {
            std::cout << "Running scenario: " << scenario.name << std::endl;
            
            // Aggregate results over replications
            std::vector<SimulationResults> rep_results;
            
            for (int rep = 0; rep < replications; rep++) {
                auto model = createModel();
                model->engine()->setSeed(base_seed + rep);
                model->configure(scenario.config);
                model->run(duration);
                rep_results.push_back(model->collectResults());
            }
            
            // Average results
            scenario.results = averageResults(rep_results);
            scenario.completed = true;
        }
    }
    
    void printComparison() const {
        std::cout << "\n" << std::string(100, '=') << "\n";
        std::cout << "SCENARIO COMPARISON REPORT\n";
        std::cout << "Business: " << business_name_ << " (" << base_model_type_ << ")\n";
        std::cout << std::string(100, '=') << "\n\n";
        
        // Header
        std::cout << std::left << std::setw(25) << "Metric";
        for (const auto& s : scenarios_) {
            std::cout << std::setw(15) << s.name;
        }
        std::cout << "\n" << std::string(100, '-') << "\n";
        
        // Metrics
        printRow("Total Customers", [](const auto& r) { return std::to_string(r.total_customers); });
        printRow("Served", [](const auto& r) { return std::to_string(r.served_customers); });
        printRow("Abandoned", [](const auto& r) { return std::to_string(r.abandoned_customers); });
        printRow("Abandonment %", [](const auto& r) { 
            return formatPercent(r.abandonment_rate * 100); });
        
        std::cout << std::string(100, '-') << "\n";
        
        printRow("Avg Wait (min)", [](const auto& r) { return formatDouble(r.avg_wait_time); });
        printRow("Max Wait (min)", [](const auto& r) { return formatDouble(r.max_wait_time); });
        printRow("P95 Wait (min)", [](const auto& r) { return formatDouble(r.p95_wait_time); });
        printRow("Avg Total Time", [](const auto& r) { return formatDouble(r.avg_total_time); });
        
        std::cout << std::string(100, '-') << "\n";
        
        printRow("Staff Utilization %", [](const auto& r) { 
            return formatPercent(r.avg_staff_utilization * 100); });
        printRow("Station Utilization %", [](const auto& r) { 
            return formatPercent(r.avg_station_utilization * 100); });
        printRow("Throughput/hr", [](const auto& r) { return formatDouble(r.customers_per_hour); });
        
        std::cout << std::string(100, '-') << "\n";
        
        printRow("Total Revenue $", [](const auto& r) { return formatCurrency(r.total_revenue); });
        printRow("Lost Revenue $", [](const auto& r) { return formatCurrency(r.lost_revenue); });
        
        std::cout << std::string(100, '-') << "\n";
        
        printRow("Bottleneck", [](const auto& r) { return r.primary_bottleneck; });
        printRow("Bottleneck Util %", [](const auto& r) { 
            return formatPercent(r.bottleneck_utilization * 100); });
        
        std::cout << "\n" << std::string(100, '=') << "\n";
    }
    
    const std::vector<Scenario>& scenarios() const { return scenarios_; }
    
private:
    std::unique_ptr<BusinessModel> createModel() {
        if (base_model_type_ == "restaurant") {
            return std::make_unique<RestaurantModel>(business_name_);
        } else if (base_model_type_ == "retail") {
            return std::make_unique<RetailStoreModel>(business_name_);
        } else if (base_model_type_ == "warehouse") {
            return std::make_unique<WarehouseModel>(business_name_);
        }
        return std::make_unique<RestaurantModel>(business_name_);
    }
    
    SimulationResults averageResults(const std::vector<SimulationResults>& results) {
        if (results.empty()) return {};
        if (results.size() == 1) return results[0];
        
        SimulationResults avg = results[0];
        int n = results.size();
        
        for (size_t i = 1; i < results.size(); i++) {
            const auto& r = results[i];
            avg.total_customers += r.total_customers;
            avg.served_customers += r.served_customers;
            avg.abandoned_customers += r.abandoned_customers;
            avg.avg_wait_time += r.avg_wait_time;
            avg.max_wait_time = std::max(avg.max_wait_time, r.max_wait_time);
            avg.p95_wait_time += r.p95_wait_time;
            avg.avg_service_time += r.avg_service_time;
            avg.avg_total_time += r.avg_total_time;
            avg.avg_staff_utilization += r.avg_staff_utilization;
            avg.avg_station_utilization += r.avg_station_utilization;
            avg.total_revenue += r.total_revenue;
            avg.lost_revenue += r.lost_revenue;
            avg.customers_per_hour += r.customers_per_hour;
        }
        
        avg.total_customers /= n;
        avg.served_customers /= n;
        avg.abandoned_customers /= n;
        avg.abandonment_rate = avg.total_customers > 0 ? 
            static_cast<double>(avg.abandoned_customers) / avg.total_customers : 0;
        avg.avg_wait_time /= n;
        avg.p95_wait_time /= n;
        avg.avg_service_time /= n;
        avg.avg_total_time /= n;
        avg.avg_staff_utilization /= n;
        avg.avg_station_utilization /= n;
        avg.total_revenue /= n;
        avg.lost_revenue /= n;
        avg.customers_per_hour /= n;
        
        return avg;
    }
    
    void printRow(const std::string& label, 
                  std::function<std::string(const SimulationResults&)> getter) const {
        std::cout << std::left << std::setw(25) << label;
        for (const auto& s : scenarios_) {
            std::cout << std::setw(15) << getter(s.results);
        }
        std::cout << "\n";
    }
    
    static std::string formatDouble(double v) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << v;
        return ss.str();
    }
    
    static std::string formatPercent(double v) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(1) << v << "%";
        return ss.str();
    }
    
    static std::string formatCurrency(double v) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(0) << v;
        return ss.str();
    }
};

// ============================================================================
// OPTIMIZER
// ============================================================================

class StaffingOptimizer {
public:
    struct Recommendation {
        std::string description;
        std::string parameter;
        int current_value;
        int recommended_value;
        double expected_improvement_percent;
        double estimated_cost_change;
    };
    
    static std::vector<Recommendation> analyze(const SimulationResults& results,
                                                const Config& current_config) {
        std::vector<Recommendation> recommendations;
        
        // Check for high abandonment rate
        if (results.abandonment_rate > 0.05) {
            recommendations.push_back({
                "High abandonment rate detected. Consider adding staff.",
                "staff_count",
                0, 1,
                results.abandonment_rate * 100,
                50.0  // Estimated hourly cost increase
            });
        }
        
        // Check bottleneck utilization
        if (results.bottleneck_utilization > 0.85) {
            recommendations.push_back({
                "Bottleneck detected at " + results.primary_bottleneck + 
                ". Consider adding capacity.",
                results.primary_bottleneck + "_servers",
                0, 1,
                (results.bottleneck_utilization - 0.75) * 100,
                75.0
            });
        }
        
        // Check wait times
        if (results.avg_wait_time > 10.0) {
            recommendations.push_back({
                "Average wait time exceeds 10 minutes. Customer satisfaction at risk.",
                "staff_count",
                0, 1,
                std::min(50.0, (results.avg_wait_time - 5.0) * 5),
                60.0
            });
        }
        
        // Check staff utilization
        if (results.avg_staff_utilization < 0.5) {
            recommendations.push_back({
                "Low staff utilization. Consider reducing staff during slow periods.",
                "staff_count",
                0, -1,
                (0.7 - results.avg_staff_utilization) * 100,
                -40.0  // Cost savings
            });
        }
        
        return recommendations;
    }
    
    static void printRecommendations(const std::vector<Recommendation>& recs) {
        std::cout << "\n📊 OPTIMIZATION RECOMMENDATIONS\n";
        std::cout << std::string(60, '-') << "\n";
        
        if (recs.empty()) {
            std::cout << "✅ Current configuration appears optimal!\n";
            return;
        }
        
        for (size_t i = 0; i < recs.size(); i++) {
            const auto& r = recs[i];
            std::cout << "\n" << (i + 1) << ". " << r.description << "\n";
            std::cout << "   Expected Improvement: " << std::fixed 
                      << std::setprecision(1) << r.expected_improvement_percent << "%\n";
            std::cout << "   Cost Impact: $" << std::setprecision(0) 
                      << r.estimated_cost_change << "/hour\n";
        }
        std::cout << "\n";
    }
};

// ============================================================================
// REPORT GENERATOR
// ============================================================================

class ReportGenerator {
public:
    static void printDetailedReport(const SimulationResults& results,
                                     const std::string& business_name) {
        std::cout << "\n";
        printBanner("DIGITAL SIMULATION REPORT");
        std::cout << "Business: " << business_name << "\n";
        std::cout << "Simulation Duration: " << results.total_simulation_time << " minutes\n";
        std::cout << std::string(70, '=') << "\n\n";
        
        // Customer Summary
        printSection("CUSTOMER FLOW");
        printMetric("Total Arrivals", results.total_customers);
        printMetric("Customers Served", results.served_customers);
        printMetric("Customers Abandoned", results.abandoned_customers);
        printMetric("Abandonment Rate", formatPercent(results.abandonment_rate));
        printMetric("Throughput", formatDouble(results.customers_per_hour) + " customers/hour");
        
        // Wait Times
        printSection("WAIT TIME ANALYSIS");
        printMetric("Average Wait", formatMinutes(results.avg_wait_time));
        printMetric("Maximum Wait", formatMinutes(results.max_wait_time));
        printMetric("95th Percentile Wait", formatMinutes(results.p95_wait_time));
        printMetric("Average Total Time", formatMinutes(results.avg_total_time));
        
        // Resource Utilization
        printSection("RESOURCE UTILIZATION");
        printMetric("Avg Staff Utilization", formatPercent(results.avg_staff_utilization));
        printMetric("Avg Station Utilization", formatPercent(results.avg_station_utilization));
        
        // Financial
        printSection("FINANCIAL IMPACT");
        printMetric("Total Revenue", "$" + formatDouble(results.total_revenue));
        printMetric("Lost Revenue (Abandonment)", "$" + formatDouble(results.lost_revenue));
        
        // Bottleneck
        printSection("BOTTLENECK ANALYSIS");
        printMetric("Primary Bottleneck", results.primary_bottleneck);
        printMetric("Bottleneck Utilization", formatPercent(results.bottleneck_utilization));
        
        // Station Details
        if (!results.station_metrics.empty()) {
            printSection("STATION DETAILS");
            std::cout << std::left 
                      << std::setw(20) << "Station"
                      << std::setw(12) << "Util %"
                      << std::setw(12) << "Avg Queue"
                      << std::setw(12) << "Avg Wait"
                      << std::setw(10) << "Served"
                      << "\n";
            std::cout << std::string(66, '-') << "\n";
            
            for (const auto& sm : results.station_metrics) {
                std::cout << std::left
                          << std::setw(20) << sm.name
                          << std::setw(12) << formatPercent(sm.utilization)
                          << std::setw(12) << formatDouble(sm.avg_queue_length)
                          << std::setw(12) << formatMinutes(sm.avg_wait_time)
                          << std::setw(10) << sm.served
                          << "\n";
            }
        }
        
        // Staff Details
        if (!results.staff_metrics.empty()) {
            printSection("STAFF DETAILS");
            std::cout << std::left
                      << std::setw(20) << "Staff Member"
                      << std::setw(15) << "Utilization %"
                      << std::setw(15) << "Customers Served"
                      << "\n";
            std::cout << std::string(50, '-') << "\n";
            
            for (const auto& sm : results.staff_metrics) {
                std::cout << std::left
                          << std::setw(20) << sm.name
                          << std::setw(15) << formatPercent(sm.utilization)
                          << std::setw(15) << sm.customers_served
                          << "\n";
            }
        }
        
        std::cout << "\n" << std::string(70, '=') << "\n";
    }
    
    static std::string generateJSON(const SimulationResults& results) {
        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"summary\": {\n";
        ss << "    \"total_customers\": " << results.total_customers << ",\n";
        ss << "    \"served_customers\": " << results.served_customers << ",\n";
        ss << "    \"abandoned_customers\": " << results.abandoned_customers << ",\n";
        ss << "    \"abandonment_rate\": " << results.abandonment_rate << ",\n";
        ss << "    \"customers_per_hour\": " << results.customers_per_hour << "\n";
        ss << "  },\n";
        ss << "  \"wait_times\": {\n";
        ss << "    \"average\": " << results.avg_wait_time << ",\n";
        ss << "    \"maximum\": " << results.max_wait_time << ",\n";
        ss << "    \"p95\": " << results.p95_wait_time << "\n";
        ss << "  },\n";
        ss << "  \"utilization\": {\n";
        ss << "    \"staff\": " << results.avg_staff_utilization << ",\n";
        ss << "    \"stations\": " << results.avg_station_utilization << "\n";
        ss << "  },\n";
        ss << "  \"financial\": {\n";
        ss << "    \"total_revenue\": " << results.total_revenue << ",\n";
        ss << "    \"lost_revenue\": " << results.lost_revenue << "\n";
        ss << "  },\n";
        ss << "  \"bottleneck\": {\n";
        ss << "    \"station\": \"" << results.primary_bottleneck << "\",\n";
        ss << "    \"utilization\": " << results.bottleneck_utilization << "\n";
        ss << "  }\n";
        ss << "}\n";
        return ss.str();
    }
    
private:
    static void printBanner(const std::string& title) {
        std::cout << std::string(70, '=') << "\n";
        std::cout << "  " << title << "\n";
        std::cout << std::string(70, '=') << "\n";
    }
    
    static void printSection(const std::string& title) {
        std::cout << "\n📌 " << title << "\n";
        std::cout << std::string(40, '-') << "\n";
    }
    
    template<typename T>
    static void printMetric(const std::string& name, T value) {
        std::cout << "  " << std::left << std::setw(25) << name << ": " << value << "\n";
    }
    
    static std::string formatDouble(double v) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << v;
        return ss.str();
    }
    
    static std::string formatPercent(double v) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(1) << (v * 100) << "%";
        return ss.str();
    }
    
    static std::string formatMinutes(double minutes) {
        if (minutes < 1) {
            return formatDouble(minutes * 60) + " sec";
        }
        return formatDouble(minutes) + " min";
    }
};

// ============================================================================
// VISUALIZATION (ASCII Charts)
// ============================================================================

class ASCIIChart {
public:
    static void printBarChart(const std::string& title,
                              const std::vector<std::pair<std::string, double>>& data,
                              int width = 50) {
        std::cout << "\n" << title << "\n" << std::string(title.length(), '-') << "\n";
        
        double max_val = 0;
        size_t max_label_len = 0;
        for (const auto& [label, value] : data) {
            max_val = std::max(max_val, value);
            max_label_len = std::max(max_label_len, label.length());
        }
        
        for (const auto& [label, value] : data) {
            std::cout << std::left << std::setw(max_label_len + 2) << label;
            
            int bar_len = max_val > 0 ? static_cast<int>((value / max_val) * width) : 0;
            std::cout << "[";
            std::cout << std::string(bar_len, '#');
            std::cout << std::string(width - bar_len, ' ');
            std::cout << "] ";
            std::cout << std::fixed << std::setprecision(1) << value << "\n";
        }
    }
    
    static void printTimeSeriesPreview(const std::string& title,
                                        const std::vector<std::pair<double, double>>& data,
                                        int width = 60, int height = 10) {
        if (data.empty()) return;
        
        std::cout << "\n" << title << "\n";
        
        double min_val = data[0].second, max_val = data[0].second;
        for (const auto& [t, v] : data) {
            min_val = std::min(min_val, v);
            max_val = std::max(max_val, v);
        }
        
        double range = max_val - min_val;
        if (range == 0) range = 1;
        
        // Create grid
        std::vector<std::string> grid(height, std::string(width, ' '));
        
        // Plot points
        int prev_x = -1, prev_y = -1;
        for (const auto& [t, v] : data) {
            int x = static_cast<int>((t / data.back().first) * (width - 1));
            int y = height - 1 - static_cast<int>(((v - min_val) / range) * (height - 1));
            
            x = std::clamp(x, 0, width - 1);
            y = std::clamp(y, 0, height - 1);
            
            grid[y][x] = '*';
            prev_x = x;
            prev_y = y;
        }
        
        // Print with axis
        std::cout << std::fixed << std::setprecision(1);
        for (int i = 0; i < height; i++) {
            double y_val = max_val - (static_cast<double>(i) / (height - 1)) * range;
            std::cout << std::setw(8) << y_val << " |" << grid[i] << "\n";
        }
        std::cout << std::string(9, ' ') << "+" << std::string(width, '-') << "\n";
    }
};

}  

// ============================================================================
// MAIN - DEMONSTRATION
// ============================================================================

int main() {
    using namespace DigitalSimulator ;
    
    std::cout << R"(
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║           🏢 DIGITAL BUSINESS SIMULATOR 🏢                 ║
    ║                                                                  ║
    ║     Discrete Event Simulation Engine for Small Businesses        ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    )" << "\n";

    // ========================================================================
    // DEMO 1: Restaurant Simulation with Scenario Comparison
    // ========================================================================
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "📍 DEMO 1: RESTAURANT SCENARIO ANALYSIS\n";
    std::cout << std::string(70, '=') << "\n";
    
    ScenarioManager restaurant_scenarios("restaurant", "Joe's Diner");
    
    // Baseline scenario
    Config baseline;
    baseline["num_hosts"] = 1;
    baseline["num_servers"] = 3;
    baseline["num_cooks"] = 2;
    baseline["num_cashiers"] = 1;
    baseline["arrival_rate"] = 40.0;  // 40 customers/hour
    baseline["avg_order_value"] = 25.0;
    restaurant_scenarios.addScenario("Baseline", "Current staffing", baseline);
    
    // Add one server
    Config more_servers = baseline;
    more_servers["num_servers"] = 4;
    restaurant_scenarios.addScenario("+1 Server", "Add one server", more_servers);
    
    // Add one cook
    Config more_cooks = baseline;
    more_cooks["num_cooks"] = 3;
    restaurant_scenarios.addScenario("+1 Cook", "Add one cook", more_cooks);
    
    // Rush hour scenario
    Config rush_hour = baseline;
    rush_hour["arrival_rate"] = 60.0;  // 60 customers/hour
    restaurant_scenarios.addScenario("Rush Hour", "Peak demand", rush_hour);
    
    // Rush hour with extra staff
    Config rush_staffed = rush_hour;
    rush_staffed["num_servers"] = 5;
    rush_staffed["num_cooks"] = 3;
    restaurant_scenarios.addScenario("Rush+Staff", "Rush with extra staff", rush_staffed);
    
    // Run all scenarios (8-hour simulation, 3 replications)
    std::cout << "\nRunning restaurant simulations...\n";
    auto start = std::chrono::steady_clock::now();
    restaurant_scenarios.runAll(480.0, 3);  // 480 minutes = 8 hours
    auto end = std::chrono::steady_clock::now();
    double runtime = std::chrono::duration<double>(end - start).count();
    
    std::cout << "Completed in " << std::fixed << std::setprecision(2) 
              << runtime << " seconds\n";
    
    // Print comparison
    restaurant_scenarios.printComparison();
    
    // Print detailed report for baseline
    std::cout << "\n📄 DETAILED REPORT FOR BASELINE SCENARIO:\n";
    ReportGenerator::printDetailedReport(
        restaurant_scenarios.scenarios()[0].results, 
        "Joe's Diner - Baseline"
    );
    
    // Generate recommendations
    auto recommendations = StaffingOptimizer::analyze(
        restaurant_scenarios.scenarios()[0].results,
        baseline
    );
    StaffingOptimizer::printRecommendations(recommendations);
    
    // ========================================================================
    // DEMO 2: Retail Store Analysis
    // ========================================================================
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "📍 DEMO 2: RETAIL STORE ANALYSIS\n";
    std::cout << std::string(70, '=') << "\n";
    
    ScenarioManager retail_scenarios("retail", "Fashion Outlet");
    
    Config retail_base;
    retail_base["num_checkouts"] = 3;
    retail_base["num_fitting_rooms"] = 4;
    retail_base["arrival_rate"] = 60.0;
    retail_base["avg_transaction"] = 75.0;
    retail_base["fitting_room_prob"] = 0.3;
    retail_scenarios.addScenario("Base", "Current setup", retail_base);
    
    Config retail_more_checkouts = retail_base;
    retail_more_checkouts["num_checkouts"] = 5;
    retail_scenarios.addScenario("+2 Checkout", "Add 2 checkouts", retail_more_checkouts);
    
    Config retail_busy = retail_base;
    retail_busy["arrival_rate"] = 100.0;
    retail_scenarios.addScenario("Busy Day", "100 customers/hr", retail_busy);
    
    std::cout << "\nRunning retail simulations...\n";
    retail_scenarios.runAll(480.0, 3);
    retail_scenarios.printComparison();
    
    // ========================================================================
    // DEMO 3: Warehouse Operations
    // ========================================================================
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "📍 DEMO 3: WAREHOUSE OPERATIONS\n";
    std::string(70, '=');
    
    ScenarioManager warehouse_scenarios("warehouse", "Central Distribution");
    
    Config warehouse_base;
    warehouse_base["num_dock_doors"] = 4;
    warehouse_base["num_inspectors"] = 2;
    warehouse_base["num_forklift_operators"] = 3;
    warehouse_base["num_pickers"] = 5;
    warehouse_base["num_packers"] = 3;
    warehouse_base["inbound_rate"] = 8.0;
    warehouse_base["order_rate"] = 30.0;
    warehouse_scenarios.addScenario("Base", "Current ops", warehouse_base);
    
    Config warehouse_more_pickers = warehouse_base;
    warehouse_more_pickers["num_pickers"] = 8;
    warehouse_scenarios.addScenario("+3 Pickers", "Add 3 pickers", warehouse_more_pickers);
    
    Config warehouse_peak = warehouse_base;
    warehouse_peak["order_rate"] = 50.0;
    warehouse_scenarios.addScenario("Peak Season", "50 orders/hr", warehouse_peak);
    
    std::cout << "\nRunning warehouse simulations...\n";
    warehouse_scenarios.runAll(480.0, 3);
    warehouse_scenarios.printComparison();
    
    // ========================================================================
    // DEMO 4: Single Detailed Run with Visualization
    // ========================================================================
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "📍 DEMO 4: DETAILED SINGLE SIMULATION WITH METRICS\n";
    std::cout << std::string(70, '=') << "\n";
    
    RestaurantModel detailed_sim("The Gourmet Kitchen");
    Config detailed_config;
    detailed_config["num_hosts"] = 1;
    detailed_config["num_servers"] = 4;
    detailed_config["num_cooks"] = 3;
    detailed_config["num_cashiers"] = 2;
    detailed_config["arrival_rate"] = 50.0;
    detailed_config["avg_order_value"] = 45.0;
    detailed_config["customer_patience"] = 12.0;
    
    detailed_sim.configure(detailed_config);
    
    std::cout << "\nRunning detailed 12-hour simulation...\n";
    detailed_sim.run(720.0);  // 12 hours
    
    auto detailed_results = detailed_sim.collectResults();
    ReportGenerator::printDetailedReport(detailed_results, "The Gourmet Kitchen");
    
    // Visualizations
    std::vector<std::pair<std::string, double>> utilization_data;
    for (const auto& sm : detailed_results.station_metrics) {
        utilization_data.emplace_back(sm.name, sm.utilization * 100);
    }
    ASCIIChart::printBarChart("Station Utilization (%)", utilization_data);
    
    std::vector<std::pair<std::string, double>> wait_data;
    for (const auto& sm : detailed_results.station_metrics) {
        wait_data.emplace_back(sm.name, sm.avg_wait_time);
    }
    ASCIIChart::printBarChart("Average Wait Time (minutes)", wait_data);
    
    // JSON Export
    std::cout << "\n📤 JSON EXPORT:\n";
    std::cout << std::string(40, '-') << "\n";
    std::cout << ReportGenerator::generateJSON(detailed_results);
    
    // ========================================================================
    // Summary
    // ========================================================================
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "✅ SIMULATION COMPLETE\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << R"(
    This Digital Twin Simulator demonstrates:
    
    ✓ Discrete Event Simulation (DES) engine
    ✓ Multiple business models (Restaurant, Retail, Warehouse)
    ✓ Queue theory implementation (FIFO, Priority)
    ✓ Staff scheduling and utilization tracking
    ✓ Bottleneck detection and analysis
    ✓ Monte Carlo simulation (multiple replications)
    ✓ Scenario comparison framework
    ✓ Optimization recommendations
    ✓ Comprehensive reporting
    
    🚀 Next Steps for Production:
    - Add REST API server for web integration
    - Implement database persistence
    - Add real-time visualization dashboard
    - Integrate machine learning for predictions
    - Add more business model templates
    
    )" << "\n";
    
    return 0;
}