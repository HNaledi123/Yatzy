#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <string>
#include <thread>
#include <vector>
#include <ctime>

namespace yatzy {

constexpr int NUM_CATEGORIES = 15;
constexpr int NUM_ROUNDS = NUM_CATEGORIES;
constexpr int DICE_PER_ROLL = 5;
constexpr int FACES = 6;
constexpr int ROLL_STATE_COUNT = 7776;  // 6^5

constexpr std::array<int, FACES> FACE_TO_UPPER_INDEX = {0, 1, 2, 3, 4, 5};
constexpr std::array<int, 6> OF_A_KIND_TO_INDEX = {0, 0, 6, 7, 8, 9};
constexpr int YATZY_CATEGORY_INDEX = OF_A_KIND_TO_INDEX[5];

constexpr std::array<int, NUM_CATEGORIES> CATEGORY_PRIORITY = {
    FACE_TO_UPPER_INDEX[5],
    FACE_TO_UPPER_INDEX[4],
    FACE_TO_UPPER_INDEX[3],
    FACE_TO_UPPER_INDEX[2],
    FACE_TO_UPPER_INDEX[1],
    FACE_TO_UPPER_INDEX[0],
    OF_A_KIND_TO_INDEX[5],
    OF_A_KIND_TO_INDEX[4],
    OF_A_KIND_TO_INDEX[3],
    OF_A_KIND_TO_INDEX[2],
    10,  // TwoPairs
    12,  // LargeStraight
    11,  // SmallStraight
    13,  // FullHouse
    14   // Chance
};

constexpr std::array<int, DICE_PER_ROLL> SMALL_STRAIGHT_TEMPLATE = {1, 2, 3, 4, 5};
constexpr std::array<int, DICE_PER_ROLL> LARGE_STRAIGHT_TEMPLATE = {2, 3, 4, 5, 6};

struct Tables {
    std::array<std::array<int16_t, NUM_CATEGORIES>, ROLL_STATE_COUNT> rollScores{};
    std::array<std::array<uint8_t, NUM_CATEGORIES>, ROLL_STATE_COUNT> rollSatisfied{};
    std::array<int8_t, ROLL_STATE_COUNT> keepValue{};
    std::array<uint8_t, ROLL_STATE_COUNT> keepCount{};

    constexpr Tables() : rollScores(), rollSatisfied(), keepValue(), keepCount() {}
};

static Tables gTables;

struct TablesBuilder {
    TablesBuilder() { build(); }

    static int encode_roll(const std::array<int, DICE_PER_ROLL>& dice) {
        int key = 0;
        for (int value : dice) {
            key = key * FACES + (value - 1);
        }
        return key;
    }

    static std::array<int, 7> analyze_counts(const std::array<int, DICE_PER_ROLL>& dice) {
        std::array<int, 7> counts{};
        for (int value : dice) {
            ++counts[value];
        }
        return counts;
    }

    static int score_upper(const std::array<int, 7>& counts, int face) {
        return counts[face] * face;
    }

    static int score_two_pairs(const std::array<int, DICE_PER_ROLL>& dice, const std::array<int, 7>& counts) {
        std::array<int, FACES> seen{};
        std::array<int, 2> pairs{};
        int pairCount = 0;
        for (int value : dice) {
            if (seen[value - 1]) {
                continue;
            }
            seen[value - 1] = 1;
            int cnt = counts[value];
            if (cnt == 2 || cnt == 3) {
                if (pairCount < 2) {
                    pairs[pairCount] = value;
                }
                ++pairCount;
            }
        }
        if (pairCount < 2) {
            return 0;
        }
        return pairs[0] * 2 + pairs[1] * 2;
    }

    static int score_n_of_a_kind(const std::array<int, 7>& counts, int required) {
        if (required == 5) {
            for (int face = FACES; face >= 1; --face) {
                if (counts[face] >= 5) {
                    return 50;
                }
            }
            return 0;
        }
        for (int face = FACES; face >= 1; --face) {
            if (counts[face] >= required) {
                return face * required;
            }
        }
        return 0;
    }

    static int score_straight(const std::array<int, DICE_PER_ROLL>& sortedDice, int size) {
        if (size == 1) {
            return std::equal(sortedDice.begin(), sortedDice.end(), LARGE_STRAIGHT_TEMPLATE.begin()) ? 20 : 0;
        }
        return std::equal(sortedDice.begin(), sortedDice.end(), SMALL_STRAIGHT_TEMPLATE.begin()) ? 15 : 0;
    }

    static int score_full_house(const std::array<int, 7>& counts) {
        bool hasPair = false;
        bool hasThree = false;
        int score = 0;
        for (int face = 1; face <= FACES; ++face) {
            int cnt = counts[face];
            if (cnt == 2) {
                hasPair = true;
                score += face * 2;
            } else if (cnt == 3) {
                hasThree = true;
                score += face * 3;
            }
        }
        return (hasPair && hasThree) ? score : 0;
    }

    static void build() {
        for (int idx = 0; idx < ROLL_STATE_COUNT; ++idx) {
            std::array<int, DICE_PER_ROLL> dice{};
            int key = idx;
            for (int pos = DICE_PER_ROLL - 1; pos >= 0; --pos) {
                dice[pos] = key % FACES + 1;
                key /= FACES;
            }

            auto counts = analyze_counts(dice);
            std::array<int, DICE_PER_ROLL> sorted = dice;
            std::sort(sorted.begin(), sorted.end());
            int diceSum = std::accumulate(dice.begin(), dice.end(), 0);

            std::array<int16_t, NUM_CATEGORIES> row{};
            for (int face = 1; face <= FACES; ++face) {
                row[FACE_TO_UPPER_INDEX[face - 1]] = score_upper(counts, face);
            }
            for (int required = 2; required <= 5; ++required) {
                row[OF_A_KIND_TO_INDEX[required]] = score_n_of_a_kind(counts, required);
            }
            row[10] = score_two_pairs(dice, counts);
            row[12] = score_straight(sorted, 1);
            row[11] = score_straight(sorted, 0);
            row[13] = score_full_house(counts);
            row[14] = diceSum;

            std::array<uint8_t, NUM_CATEGORIES> sat{};
            for (int category = 0; category < NUM_CATEGORIES; ++category) {
                if (row[category] > 0) {
                    sat[category] = 1;
                }
            }

            int maxCount = 0;
            int keepValue = 1;
            for (int face = 6; face >= 1; --face) {
                int cnt = counts[face];
                if (cnt > maxCount) {
                    maxCount = cnt;
                    keepValue = face;
                }
            }

            gTables.rollScores[idx] = row;
            gTables.rollSatisfied[idx] = sat;
            gTables.keepValue[idx] = static_cast<int8_t>(keepValue);
            gTables.keepCount[idx] = static_cast<uint8_t>(maxCount);
        }
    }
};

static TablesBuilder gTablesBuilder;

struct SplitMix64 {
    uint64_t state;

    explicit SplitMix64(uint64_t seed) : state(seed) {}

    uint64_t next() {
        uint64_t z = (state += UINT64_C(0x9E3779B97F4A7C15));
        z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
        return z ^ (z >> 31);
    }
};

inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

class Xoshiro256ss {
   public:
    explicit Xoshiro256ss(uint64_t seed) {
        SplitMix64 sm(seed);
        for (auto& s : state_) {
            s = sm.next();
        }
    }

    uint64_t next() {
        const uint64_t result = rotl(state_[1] * 5, 7) * 9;
        const uint64_t t = state_[1] << 17;

        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];

        state_[2] ^= t;

        state_[3] = rotl(state_[3], 45);
        return result;
    }

   private:
    std::array<uint64_t, 4> state_{};
};

inline uint8_t roll_die(Xoshiro256ss& rng) {
    const uint64_t value = rng.next();
    const uint32_t scaled = static_cast<uint32_t>((static_cast<__uint128_t>(value) * 6u) >> 64);
    return static_cast<uint8_t>(scaled + 1u);
}

inline int encode_roll(const std::array<int8_t, DICE_PER_ROLL>& dice) {
    int key = 0;
    for (int value : dice) {
        key = key * FACES + (value - 1);
    }
    return key;
}

inline void reroll_keep_most_common(const std::array<int8_t, DICE_PER_ROLL>& dice,
                                    int rollIndex,
                                    Xoshiro256ss& rng,
                                    std::array<int8_t, DICE_PER_ROLL>& out) {
    const uint8_t keepCount = gTables.keepCount[rollIndex];
    if (keepCount == DICE_PER_ROLL) {
        out = dice;
        return;
    }
    const int8_t keepValue = gTables.keepValue[rollIndex];
    int i = 0;
    for (; i < keepCount; ++i) {
        out[i] = keepValue;
    }
    for (; i < DICE_PER_ROLL; ++i) {
        out[i] = static_cast<int8_t>(roll_die(rng));
    }
}

inline std::pair<int, int> evaluate_best_category(int rollIndex, uint16_t allowedBits) {
    const auto& scores = gTables.rollScores[rollIndex];
    int bestScore = -1;
    int bestIdx = -1;
    for (int category : CATEGORY_PRIORITY) {
        if ((allowedBits >> category) & 1u) {
            int score = scores[category];
            if (score > bestScore) {
                bestScore = score;
                bestIdx = category;
            }
        }
    }
    return {bestScore, bestIdx};
}

inline uint16_t play_yatzy(Xoshiro256ss& rng) {
    uint16_t allowedBits = static_cast<uint16_t>((1u << NUM_CATEGORIES) - 1u);
    int totalPoints = 0;
    int upperPoints = 0;

    std::array<int8_t, DICE_PER_ROLL> dice0{};
    std::array<int8_t, DICE_PER_ROLL> dice1{};
    std::array<int8_t, DICE_PER_ROLL> dice2{};

    for (int round = 0; round < NUM_ROUNDS; ++round) {
        for (auto& value : dice0) {
            value = static_cast<int8_t>(roll_die(rng));
        }
        const int idx0 = encode_roll(dice0);
        reroll_keep_most_common(dice0, idx0, rng, dice1);

        const int idx1 = encode_roll(dice1);
        reroll_keep_most_common(dice1, idx1, rng, dice2);

        const int idx2 = encode_roll(dice2);
        auto [score, chosenIdx] = evaluate_best_category(idx2, allowedBits);

        totalPoints += score;
        if (chosenIdx < 6) {
            upperPoints += score;
        }
        allowedBits = static_cast<uint16_t>(allowedBits & ~(1u << chosenIdx));
    }

    if (upperPoints >= 63) {
        totalPoints += 50;
    }
    return static_cast<uint16_t>(totalPoints);
}

struct ThreadResult {
    std::vector<uint64_t> counts{};
    uint64_t gamesSimulated = 0;
};

void run_thread(uint64_t games, uint64_t seed, ThreadResult& result) {
    result.gamesSimulated = games;
    result.counts.clear();
    result.counts.resize(1, 0);
    Xoshiro256ss rng(seed);
    for (uint64_t i = 0; i < games; ++i) {
        const uint16_t score = play_yatzy(rng);
        if (score >= result.counts.size()) {
            result.counts.resize(static_cast<std::size_t>(score) + 1, 0);
        }
        ++result.counts[score];
    }
}

struct SimulationSummary {
    uint64_t totalGames = 0;
    double elapsedSeconds = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
    int minScore = 0;
    int maxScore = 0;
    std::vector<uint64_t> counts;
};

std::string format_with_commas(uint64_t value) {
    std::string digits = std::to_string(value);
    std::string result;
    result.reserve(digits.size() + digits.size() / 3);
    int group = 0;
    for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
        if (group == 3) {
            result.push_back(',');
            group = 0;
        }
        result.push_back(*it);
        ++group;
    }
    std::reverse(result.begin(), result.end());
    return result;
}

std::string format_duration(double seconds) {
    if (!std::isfinite(seconds)) {
        return "n/a";
    }
    seconds = std::max(0.0, seconds);
    const long long totalSeconds = static_cast<long long>(std::llround(seconds));
    const long long hours = totalSeconds / 3600;
    const long long minutes = (totalSeconds % 3600) / 60;
    const long long secs = totalSeconds % 60;

    std::ostringstream oss;
    if (hours > 0) {
        oss << hours << "h " << std::setw(2) << std::setfill('0') << minutes << "m " << std::setw(2)
            << secs << "s";
    } else if (minutes > 0) {
        oss << minutes << "m " << std::setw(2) << std::setfill('0') << secs << "s";
    } else {
        oss << secs << "s";
    }
    return oss.str();
}

std::string format_elapsed(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) {
        return "n/a";
    }
    if (seconds < 0.001) {
        return "0ms";
    }
    if (seconds < 1.0) {
        return std::to_string(static_cast<long long>(std::llround(seconds * 1000.0))) + "ms";
    }
    if (seconds < 60.0) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << seconds << "s";
        return oss.str();
    }
    return format_duration(seconds);
}

std::string format_completion_timestamp(double secondsFromNow) {
    if (!std::isfinite(secondsFromNow) || secondsFromNow < 0.0) {
        return {};
    }
    auto now = std::chrono::system_clock::now();
    auto future =
        now + std::chrono::milliseconds(static_cast<long long>(secondsFromNow * 1000.0));
    std::time_t futureTime = std::chrono::system_clock::to_time_t(future);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &futureTime);
#else
    localtime_r(&futureTime, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

std::string format_eta_message(uint64_t remainingGames, double overallRate) {
    if (remainingGames == 0) {
        return "complete";
    }
    if (!std::isfinite(overallRate) || overallRate <= 0.0) {
        return "estimating...";
    }
    const double etaSeconds = static_cast<double>(remainingGames) / overallRate;
    const std::string duration = format_duration(etaSeconds);
    const std::string completion = format_completion_timestamp(etaSeconds);
    if (!completion.empty()) {
        return duration + " (finish ~ " + completion + ")";
    }
    return duration;
}

std::string format_rate(double rate) {
    if (!std::isfinite(rate)) {
        return "n/a";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(0) << rate << " games/s";
    return oss.str();
}

std::string format_progress_message(uint64_t processed,
                                    uint64_t total,
                                    uint64_t batchSize,
                                    double chunkSeconds,
                                    double totalSeconds) {
    const double percent =
        total == 0 ? 100.0 : static_cast<double>(processed) / static_cast<double>(total) * 100.0;
    const std::string chunkTimeStr = format_elapsed(chunkSeconds);
    const double chunkRate =
        chunkSeconds > 0.0 ? static_cast<double>(batchSize) / chunkSeconds : std::numeric_limits<double>::infinity();
    const double overallRate =
        totalSeconds > 0.0 ? static_cast<double>(processed) / totalSeconds : std::numeric_limits<double>::infinity();
    const std::string chunkRateStr = format_rate(chunkRate);
    const std::string overallRateStr = format_rate(overallRate);
    const std::string etaMessage =
        format_eta_message(total > processed ? total - processed : 0, overallRate);

    std::ostringstream oss;
    oss << "Chunk complete: " << format_with_commas(processed) << "/"
        << format_with_commas(total) << " (" << std::fixed << std::setprecision(2) << percent
        << "%) | "
        << "chunk " << format_with_commas(batchSize) << " in " << chunkTimeStr << " ("
        << chunkRateStr << ") | "
        << "overall " << overallRateStr << " | ETA " << etaMessage;
    return oss.str();
}

SimulationSummary summarize_counts(const std::vector<uint64_t>& counts, double elapsedSeconds) {
    SimulationSummary summary;
    summary.counts = counts;
    summary.elapsedSeconds = elapsedSeconds;
    for (uint64_t freq : counts) {
        summary.totalGames += freq;
    }
    if (summary.totalGames == 0) {
        summary.minScore = 0;
        summary.maxScore = 0;
        return summary;
    }

    std::size_t minIdx = 0;
    while (minIdx < counts.size() && counts[minIdx] == 0) {
        ++minIdx;
    }
    std::size_t maxIdx = counts.size();
    while (maxIdx > 0 && counts[maxIdx - 1] == 0) {
        --maxIdx;
    }

    summary.minScore = static_cast<int>(minIdx);
    summary.maxScore = static_cast<int>(maxIdx == 0 ? 0 : maxIdx - 1);

    long double meanNumerator = 0.0L;
    long double squaredNumerator = 0.0L;
    const long double totalGamesLd = static_cast<long double>(summary.totalGames);

    for (std::size_t score = minIdx; score < maxIdx; ++score) {
        const uint64_t freq = counts[score];
        if (freq == 0) {
            continue;
        }
        const long double s = static_cast<long double>(score);
        const long double f = static_cast<long double>(freq);
        meanNumerator += s * f;
        squaredNumerator += s * s * f;
    }

    const long double meanLd = meanNumerator / totalGamesLd;
    long double varianceLd = (squaredNumerator / totalGamesLd) - (meanLd * meanLd);
    if (varianceLd < 0.0L) {
        varianceLd = 0.0L;
    }
    summary.mean = static_cast<double>(meanLd);
    summary.stddev = std::sqrt(static_cast<double>(varianceLd));
    return summary;
}

void print_summary(const SimulationSummary& summary, unsigned threadsUsed) {
    std::cout << "\n--- SCORE DISTRIBUTION SUMMARY ---\n";
    std::cout << "Games simulated: " << format_with_commas(summary.totalGames) << "\n";
    std::cout << "Elapsed time: " << format_elapsed(summary.elapsedSeconds) << "\n";
    const double throughput =
        summary.elapsedSeconds > 0.0
            ? static_cast<double>(summary.totalGames) / summary.elapsedSeconds
            : std::numeric_limits<double>::infinity();
    if (std::isfinite(throughput)) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(0) << throughput;
        std::cout << "Throughput: " << oss.str() << " games/s\n";
    } else {
        std::cout << "Throughput: n/a\n";
    }
    std::cout << "Threads used: " << threadsUsed << "\n";

    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << summary.mean;
        std::cout << "Mean score: " << oss.str() << "\n";
    }
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << summary.stddev;
        std::cout << "Std dev: " << oss.str() << "\n";
    }
    std::cout << "Score range: " << summary.minScore << " - " << summary.maxScore << "\n";

    std::vector<std::pair<uint64_t, int>> entries;
    entries.reserve(summary.counts.size());
    for (std::size_t score = 0; score < summary.counts.size(); ++score) {
        const uint64_t freq = summary.counts[score];
        if (freq == 0) {
            continue;
        }
        entries.emplace_back(freq, static_cast<int>(score));
    }

    std::sort(entries.begin(),
              entries.end(),
              [](const auto& lhs, const auto& rhs) {
                  if (lhs.first == rhs.first) {
                      return lhs.second > rhs.second;
                  }
                  return lhs.first > rhs.first;
              });

    if (!entries.empty()) {
        const std::size_t limit = std::min<std::size_t>(5, entries.size());
        std::cout << "Top " << limit << " scores by frequency:\n";
        for (std::size_t idx = 0; idx < limit; ++idx) {
            const auto& entry = entries[idx];
            const double pct =
                summary.totalGames > 0
                    ? static_cast<double>(entry.first) / static_cast<double>(summary.totalGames) *
                          100.0
                    : 0.0;
            std::ostringstream line;
            line << "  " << std::setw(3) << entry.second << ": " << format_with_commas(entry.first)
                 << " (" << std::fixed << std::setprecision(4) << pct << "%)";
            std::cout << line.str() << "\n";
        }
    }
    std::cout << std::endl;
}

std::filesystem::path save_distribution(const std::string& outputDir,
                                        const std::vector<uint64_t>& counts,
                                        std::time_t runStartUnix) {
    std::filesystem::path directory(outputDir);
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec) {
        throw std::runtime_error("Failed to create output directory '" + directory.string() +
                                 "': " + ec.message());
    }

    std::ostringstream filename;
    filename << "yatzy_distribution_" << runStartUnix << ".csv";
    const std::filesystem::path filePath = directory / filename.str();

    std::ofstream file(filePath);
    if (!file) {
        throw std::runtime_error("Failed to open output file: " + filePath.string());
    }

    file << "score,count\n";
    for (std::size_t score = 0; score < counts.size(); ++score) {
        const uint64_t freq = counts[score];
        if (freq == 0) {
            continue;
        }
        file << score << "," << freq << "\n";
    }
    return filePath;
}

}  // namespace yatzy

int main(int argc, char** argv) {
    using namespace yatzy;

    const std::string programName = (argc > 0 && argv[0] != nullptr) ? argv[0] : "YatzyUltra";

    auto print_usage = [&](std::ostream& out) {
        out << "Usage: " << programName << " [options]\n\n";
        out << "Options:\n";
        out << "  --count N                  Number of games to simulate (default: 1,000,000)\n";
        out << "  --threads T                Number of worker threads (default: hardware concurrency)\n";
        out << "  --seed S                   RNG seed (default: random)\n";
        out << "  --chunk-size VALUE         Games per progress batch (default: auto; accepts 'auto' or 'none')\n";
        out << "  --output-dir PATH          Directory for CSV distribution (default: results)\n";
        out << "  --no-save-distribution     Skip writing the score distribution CSV\n";
        out << "  --save-distribution        Force writing the score distribution CSV\n";
        out << "  --help                     Show this help message\n";
    };

    auto parse_positive_uint64 = [](const std::string& value, const char* name) -> uint64_t {
        if (value.empty()) {
            throw std::invalid_argument(std::string("Missing value for ") + name);
        }
        std::size_t pos = 0;
        uint64_t parsed = 0;
        try {
            parsed = std::stoull(value, &pos);
        } catch (const std::exception&) {
            throw std::invalid_argument(std::string("Invalid value for ") + name + ": " + value);
        }
        if (pos != value.size() || parsed == 0) {
            throw std::invalid_argument(std::string("Invalid value for ") + name + ": " + value);
        }
        return parsed;
    };

    auto parse_seed = [](const std::string& value) -> uint64_t {
        if (value.empty()) {
            throw std::invalid_argument("Missing value for --seed");
        }
        std::size_t pos = 0;
        uint64_t parsed = 0;
        try {
            parsed = std::stoull(value, &pos);
        } catch (const std::exception&) {
            throw std::invalid_argument("Invalid value for --seed: " + value);
        }
        if (pos != value.size()) {
            throw std::invalid_argument("Invalid value for --seed: " + value);
        }
        return parsed;
    };

    try {
        uint64_t totalGames = 1'000'000;
        unsigned requestedThreads = std::thread::hardware_concurrency();
        if (requestedThreads == 0) {
            requestedThreads = 1;
        }
        uint64_t seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::optional<uint64_t> chunkSizeOpt;
        std::string outputDir = "results";
        bool saveDistribution = true;

        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);

            auto match_option = [&](const std::string& opt) -> std::optional<std::string> {
                const std::string prefix = opt + "=";
                if (arg == opt) {
                    if (i + 1 >= argc) {
                        throw std::invalid_argument(opt + " requires a value");
                    }
                    return std::string(argv[++i]);
                }
                if (arg.rfind(prefix, 0) == 0) {
                    return arg.substr(prefix.size());
                }
                return std::nullopt;
            };

            if (arg == "--help") {
                print_usage(std::cout);
                return 0;
            } else if (auto value = match_option("--count")) {
                totalGames = parse_positive_uint64(*value, "--count");
            } else if (auto value = match_option("--threads")) {
                uint64_t parsed = parse_positive_uint64(*value, "--threads");
                if (parsed > static_cast<uint64_t>(std::numeric_limits<unsigned>::max())) {
                    throw std::invalid_argument("Value for --threads is too large");
                }
                requestedThreads = static_cast<unsigned>(parsed);
            } else if (auto value = match_option("--seed")) {
                seed = parse_seed(*value);
            } else if (auto value = match_option("--chunk-size")) {
                std::string lower = *value;
                std::transform(lower.begin(),
                               lower.end(),
                               lower.begin(),
                               [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
                if (lower == "auto" || lower == "none") {
                    chunkSizeOpt.reset();
                } else {
                    chunkSizeOpt = parse_positive_uint64(*value, "--chunk-size");
                }
            } else if (auto value = match_option("--output-dir")) {
                outputDir = *value;
                if (outputDir.empty()) {
                    throw std::invalid_argument("Output directory may not be empty");
                }
            } else if (arg == "--no-save-distribution") {
                saveDistribution = false;
            } else if (arg == "--save-distribution") {
                saveDistribution = true;
            } else {
                throw std::invalid_argument("Unknown argument: " + arg);
            }
        }

        if (totalGames == 0) {
            throw std::invalid_argument("--count must be greater than zero");
        }
        const unsigned numThreads = std::max(1u, requestedThreads);

        const uint64_t chunkSizeDefault = std::min<uint64_t>(totalGames, 10'000'000ULL);
        uint64_t chunkSize = chunkSizeOpt.value_or(chunkSizeDefault);
        if (chunkSize == 0) {
            chunkSize = 1;
        }

        std::vector<uint64_t> globalCounts(1, 0);
        std::vector<ThreadResult> threadResults(numThreads);
        std::vector<std::thread> workers;
        std::vector<unsigned> activeIndices;
        workers.reserve(numThreads);
        activeIndices.reserve(numThreads);

        SplitMix64 seeder(seed);
        const auto startTime = std::chrono::steady_clock::now();
        const auto runStartWallClock = std::chrono::system_clock::now();

        uint64_t processed = 0;
        while (processed < totalGames) {
            const uint64_t remaining = totalGames - processed;
            const uint64_t batchSize = std::min(chunkSize, remaining);
            const auto chunkStart = std::chrono::steady_clock::now();

            workers.clear();
            activeIndices.clear();

            const uint64_t baseGames = batchSize / numThreads;
            uint64_t remainder = batchSize % numThreads;

            for (unsigned threadIdx = 0; threadIdx < numThreads; ++threadIdx) {
                uint64_t games = baseGames + (threadIdx < remainder ? 1 : 0);
                if (games == 0) {
                    threadResults[threadIdx].gamesSimulated = 0;
                    threadResults[threadIdx].counts.clear();
                    continue;
                }
                const uint64_t threadSeed = seeder.next();
                workers.emplace_back(run_thread,
                                     games,
                                     threadSeed,
                                     std::ref(threadResults[threadIdx]));
                activeIndices.push_back(threadIdx);
            }

            for (auto& worker : workers) {
                worker.join();
            }

            for (unsigned idx : activeIndices) {
                const auto& result = threadResults[idx];
                if (result.counts.empty()) {
                    continue;
                }
                if (globalCounts.size() < result.counts.size()) {
                    globalCounts.resize(result.counts.size(), 0);
                }
                for (std::size_t score = 0; score < result.counts.size(); ++score) {
                    globalCounts[score] += result.counts[score];
                }
            }

            processed += batchSize;
            const auto chunkEnd = std::chrono::steady_clock::now();
            const double chunkSeconds =
                std::chrono::duration<double>(chunkEnd - chunkStart).count();
            const double totalSeconds =
                std::chrono::duration<double>(chunkEnd - startTime).count();
            std::cout << format_progress_message(processed,
                                                 totalGames,
                                                 batchSize,
                                                 chunkSeconds,
                                                 totalSeconds)
                      << std::endl;
        }

        const auto endTime = std::chrono::steady_clock::now();
        const double elapsedSeconds =
            std::chrono::duration<double>(endTime - startTime).count();

        SimulationSummary summary = summarize_counts(globalCounts, elapsedSeconds);
        print_summary(summary, numThreads);

        if (saveDistribution) {
            const std::time_t runStartUnix =
                std::chrono::system_clock::to_time_t(runStartWallClock);
            const auto path = save_distribution(outputDir, summary.counts, runStartUnix);
            std::cout << "Score distribution saved to: " << path.string() << std::endl;
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        std::cerr << "Use --help to see available options.\n";
        return 1;
    }
}
