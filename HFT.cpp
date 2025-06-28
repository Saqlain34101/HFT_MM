#include <iostream>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>
#include <array>
#include <immintrin.h>  // For SIMD intrinsics
#include <cstring>      // For memcpy
#include <span>

// Compiler hints for inlining and optimization
#define ALWAYS_INLINE inline __attribute__((always_inline))
#define HOT __attribute__((hot))
#define COLD __attribute__((cold))

namespace hft {

// Constants
constexpr int MAX_SYMBOLS = 100;
constexpr int ORDER_ID_SIZE = 16;
constexpr int PRICE_SCALE = 10000;
constexpr double RISK_LIMIT = 1000000.0;
constexpr int MAX_ORDER_AGE_MS = 100;

// Type aliases for performance
using OrderID = std::array<char, ORDER_ID_SIZE>;
using Timestamp = uint64_t;
using Price = int64_t;
using Quantity = int32_t;

// Packed structs for cache efficiency
#pragma pack(push, 1)
struct Order {
    OrderID id;
    Price price;
    Quantity quantity;
    bool is_bid;
    Timestamp timestamp;
    uint16_t symbol_index;
};
#pragma pack(pop)

struct MarketData {
    Price bid_price;
    Price ask_price;
    Quantity bid_quantity;
    Quantity ask_quantity;
    Timestamp last_update;
};

struct Position {
    double quantity;
    double pnl;
};

// Global state - aligned for cache lines
struct alignas(64) TradingState {
    std::array<MarketData, MAX_SYMBOLS> market_data;
    std::unordered_map<OrderID, Order> active_orders;
    std::array<Position, MAX_SYMBOLS> positions;
    std::atomic<double> total_pnl;
    std::atomic<double> total_volume;
};

// Fast memory pool for order allocation
class OrderPool {
    static constexpr size_t POOL_SIZE = 65536;
    std::array<Order, POOL_SIZE> memory;
    std::atomic<size_t> index{0};

public:
    ALWAYS_INLINE HOT Order* allocate() {
        size_t i = index.fetch_add(1, std::memory_order_relaxed) % POOL_SIZE;
        return &memory[i];
    }
};

// Low-latency trading engine
class TradingEngine {
    TradingState& state;
    OrderPool& order_pool;
    
    // Spread calculation parameters
    double min_spread = 0.0002;  // 2 basis points
    double volatility_multiplier = 2.0;
    double inventory_penalty = 0.0001;
    
    // SIMD optimized market data processing
    ALWAYS_INLINE HOT void update_market_data_simd(uint16_t symbol_index, 
                                                 const MarketData& new_data) {
        __m128i new_bid = _mm_set1_epi64x(new_data.bid_price);
        __m128i new_ask = _mm_set1_epi64x(new_data.ask_price);
        
        MarketData& current = state.market_data[symbol_index];
        __m128i current_bid = _mm_set1_epi64x(current.bid_price);
        __m128i current_ask = _mm_set1_epi64x(current.ask_price);
        
        // Compare and update only if new data is more recent
        if (new_data.last_update > current.last_update) {
            _mm_store_si128(reinterpret_cast<__m128i*>(&current.bid_price), new_bid);
            _mm_store_si128(reinterpret_cast<__m128i*>(&current.ask_price), new_ask);
            current.bid_quantity = new_data.bid_quantity;
            current.ask_quantity = new_data.ask_quantity;
            current.last_update = new_data.last_update;
        }
    }
    
    // Fast order generation
    ALWAYS_INLINE HOT Order* create_order(uint16_t symbol_index, Price price, 
                                        Quantity quantity, bool is_bid) {
        Order* order = order_pool.allocate();
        
        // Generate order ID (in practice, this would be more sophisticated)
        static std::atomic<uint64_t> counter{0};
        uint64_t id_num = counter.fetch_add(1, std::memory_order_relaxed);
        std::memcpy(order->id.data(), &id_num, sizeof(id_num));
        std::memset(order->id.data() + sizeof(id_num), 0, ORDER_ID_SIZE - sizeof(id_num));
        
        order->price = price;
        order->quantity = quantity;
        order->is_bid = is_bid;
        order->timestamp = get_timestamp();
        order->symbol_index = symbol_index;
        
        return order;
    }
    
    // Optimal spread calculation
    ALWAYS_INLINE HOT double calculate_spread(uint16_t symbol_index) {
        const MarketData& md = state.market_data[symbol_index];
        const Position& pos = state.positions[symbol_index];
        
        // Basic spread components
        double mid_price = (md.bid_price + md.ask_price) / (2.0 * PRICE_SCALE);
        double spread = min_spread + (volatility_multiplier * calculate_volatility(symbol_index));
        
        // Inventory adjustment
        double inventory_factor = 1.0 - (inventory_penalty * pos.quantity);
        spread *= inventory_factor;
        
        return std::max(spread, min_spread);
    }
    
    // Placeholder for volatility calculation
    ALWAYS_INLINE HOT double calculate_volatility(uint16_t symbol_index) {
        // In practice, this would use recent price movements
        return 0.0005;  // 5 basis points as placeholder
    }
    
    // Risk check - branchless implementation
    ALWAYS_INLINE HOT bool check_risk(uint16_t symbol_index, Quantity quantity, bool is_bid) {
        double notional = (is_bid ? state.market_data[symbol_index].ask_price : 
                                  state.market_data[symbol_index].bid_price) / 
                         static_cast<double>(PRICE_SCALE) * quantity;
        
        double potential_pnl = state.total_pnl.load(std::memory_order_relaxed);
        potential_pnl += is_bid ? -notional : notional;
        
        // Branchless return
        return (potential_pnl > -RISK_LIMIT) & (potential_pnl < RISK_LIMIT);
    }
    
    // Fast timestamp (would use platform-specific high-resolution timer in practice)
    ALWAYS_INLINE HOT Timestamp get_timestamp() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
    
public:
    TradingEngine(TradingState& s, OrderPool& op) : state(s), order_pool(op) {}
    
    // Process market data update
    ALWAYS_INLINE HOT void on_market_data(uint16_t symbol_index, const MarketData& data) {
        update_market_data_simd(symbol_index, data);
    }
    
    // Generate market making quotes
    ALWAYS_INLINE HOT std::pair<Order*, Order*> generate_quotes(uint16_t symbol_index) {
        const MarketData& md = state.market_data[symbol_index];
        double spread = calculate_spread(symbol_index);
        double mid_price = (md.bid_price + md.ask_price) / (2.0 * PRICE_SCALE);
        
        // Calculate bid/ask prices
        Price bid_price = static_cast<Price>((mid_price - spread/2) * PRICE_SCALE);
        Price ask_price = static_cast<Price>((mid_price + spread/2) * PRICE_SCALE);
        
        // Clip to tick size (assuming 1 tick = 1/PRICE_SCALE)
        bid_price = std::max(bid_price, md.bid_price - 10 * PRICE_SCALE/10000);
        ask_price = std::min(ask_price, md.ask_price + 10 * PRICE_SCALE/10000);
        
        // Fixed quantity for simplicity (could be dynamic)
        Quantity quantity = 100;
        
        // Check risk limits
        bool can_bid = check_risk(symbol_index, quantity, true);
        bool can_ask = check_risk(symbol_index, quantity, false);
        
        Order* bid_order = can_bid ? create_order(symbol_index, bid_price, quantity, true) : nullptr;
        Order* ask_order = can_ask ? create_order(symbol_index, ask_price, quantity, false) : nullptr;
        
        return {bid_order, ask_order};
    }
    
    // Handle order execution
    ALWAYS_INLINE HOT void on_execution(const OrderID& order_id, Quantity filled_quantity, Price fill_price) {
        auto it = state.active_orders.find(order_id);
        if (it != state.active_orders.end()) {
            const Order& order = it->second;
            double fill_notional = (fill_price / static_cast<double>(PRICE_SCALE)) * filled_quantity;
            
            // Update position and PnL
            Position& pos = state.positions[order.symbol_index];
            if (order.is_bid) {
                pos.quantity += filled_quantity;
                state.total_pnl.fetch_sub(fill_notional, std::memory_order_relaxed);
            } else {
                pos.quantity -= filled_quantity;
                state.total_pnl.fetch_add(fill_notional, std::memory_order_relaxed);
            }
            
            state.total_volume.fetch_add(fill_notional, std::memory_order_relaxed);
            state.active_orders.erase(it);
        }
    }
    
    // Cancel stale orders
    ALWAYS_INLINE HOT void cancel_stale_orders() {
        Timestamp now = get_timestamp();
        std::vector<OrderID> to_cancel;
        
        for (const auto& [id, order] : state.active_orders) {
            if ((now - order.timestamp) > MAX_ORDER_AGE_MS * 1'000'000) {
                to_cancel.push_back(id);
            }
        }
        
        for (const auto& id : to_cancel) {
            state.active_orders.erase(id);
            // In practice, would send cancel to exchange
        }
    }
};

// Network processing thread
void network_processing_thread(TradingEngine& engine) {
    // In practice, this would use kernel bypass (like DPDK) or specialized API
    while (true) {
        // Simulate receiving market data
        MarketData md;
        // ... populate md from network ...
        
        // Process update
        engine.on_market_data(0, md);
        
        // Yield to avoid spinning (in real HFT, would use proper busy-wait or interrupts)
        std::this_thread::yield();
    }
}

// Trading decision thread
void trading_thread(TradingEngine& engine) {
    // Set thread affinity and priority (Linux example)
    // cpu_set_t cpuset;
    // CPU_ZERO(&cpuset);
    // CPU_SET(2, &cpuset);  // Pin to core 2
    // pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    // struct sched_param param = { .sched_priority = 99 };
    // pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    
    while (true) {
        // Generate quotes
        auto [bid_order, ask_order] = engine.generate_quotes(0);
        
        // Process orders (in practice, would send to exchange)
        if (bid_order) {
            engine.state.active_orders[bid_order->id] = *bid_order;
        }
        if (ask_order) {
            engine.state.active_orders[ask_order->id] = *ask_order;
        }
        
        // Cancel stale orders
        engine.cancel_stale_orders();
        
        // Adaptive sleep based on market conditions
        static int sleep_us = 10;  // Start with 10μs
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        
        // Adaptive adjustment (simplified)
        sleep_us = std::clamp(sleep_us + (rand() % 3 - 1), 1, 100);
    }
}

}  // namespace hft

int main() {
    hft::TradingState state;
    hft::OrderPool order_pool;
    hft::TradingEngine engine(state, order_pool);
    
    // Start network thread
    std::thread network_thread(hft::network_processing_thread, std::ref(engine));
    
    // Start trading thread
    std::thread trading_thread(hft::trading_thread, std::ref(engine));
    
    // Monitor performance
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        double pnl = state.total_pnl.load();
        double volume = state.total_volume.load();
        std::cout << "PNL: " << pnl << " Volume: " << volume 
                  << " PnL/Volume: " << (volume > 0 ? pnl/volume : 0) << std::endl;
    }
    
    network_thread.join();
    trading_thread.join();
    
    return 0;
}
