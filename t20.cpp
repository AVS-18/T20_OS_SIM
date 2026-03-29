#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include <time.h>
#include <stdio.h>
#include <random>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <iomanip>
#include <sstream>
#include <atomic>
#include <chrono> 
#include <fstream>

// --- SCHEDULER CONFIGURATION ---
std::string bowler_scheduler_type = "rr"; // Default
std::string batter_scheduler_type = "fcfs"; // Default (Only option for now)

// --- GLOBAL FILE STREAMS ---
std::ofstream comm_log;
std::ofstream gantt_file;
std::ofstream wait_log;

// ============================================================
// GANTT CHART INFRASTRUCTURE (FULL GRANULARITY)
// ============================================================
struct GanttRecord {
    std::string thread_type;
    std::string player_name;
    double start_time;
    double end_time;
    std::string action;
};

std::vector<GanttRecord> gantt_log;
pthread_mutex_t gantt_mutex = PTHREAD_MUTEX_INITIALIZER;
std::chrono::steady_clock::time_point sim_start_time;

double get_sim_time() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - sim_start_time).count();
}

void log_gantt(const std::string& type, const std::string& name, double start, double end, const std::string& action) {
    pthread_mutex_lock(&gantt_mutex);
    gantt_log.push_back({type, name, start, end, action});
    pthread_mutex_unlock(&gantt_mutex);
}

const double TIME_SCALE = 0.05; 

// --- FOR DEADLOCK LOGIC (OS WAIT-FOR GRAPH) ---
std::atomic<int> alloc_graph[2]; 
std::atomic<int> req_graph[2];   

// ============================================================
// MATH & RANDOM ENGINE
// ============================================================
class RandomEngine {
private:
    std::mt19937 rng;
    std::normal_distribution<double>       normal_dist;
    std::uniform_real_distribution<double> uniform_dist;

public:
    RandomEngine()
        : rng(std::random_device{}()),
          normal_dist(0.0, 1.0),
          uniform_dist(0.0, 1.0) {}

    double gaussian(double mean, double stddev) {
        return mean + stddev * normal_dist(rng);
    }
    double uniform(double lo, double hi) {
        return lo + (hi - lo) * uniform_dist(rng);
    }
    int categorical(const std::vector<double>& probs) {
        double roll = uniform_dist(rng);
        double cum  = 0.0;
        for (size_t i = 0; i < probs.size(); i++) {
            cum += probs[i];
            if (roll < cum) return (int)i;
        }
        return (int)probs.size() - 1;
    }
};

inline thread_local RandomEngine threadRng;

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// ============================================================
// ENUMS & STRUCTS
// ============================================================
enum class Crease_End  { END_1, END_2 };
enum class Player_Role { BATSMAN, BOWLER, FIELDER };
enum class BallType    { NORMAL, WIDE, NO_BALL, CREASE_SWAP, RUNNING_PHASE }; 
enum class HitOutcome  { MISS, LBW, CAUGHT, SIX, FOUR, CONNECT };

struct SkillProfile {
    double technique, power, running_speed,
           pace, accuracy,
           catching_skill, agility, throwing_accuracy;
};

struct BatStats {
    int runs = 0; int balls = 0; int fours = 0; int sixes = 0;
    bool has_batted = false; bool is_out = false;
    std::string dismissal = "not out";
    double arrival_time = 0.0;
    double wait_time = 0.0;
};

struct BowlStats {
    int balls = 0; int runs = 0; int wickets = 0; int wides = 0; int no_balls = 0;
};

typedef struct {
    Player_Role  current_role;
    std::string  player_name;
    std::string  player_team;
    int          player_id;
    SkillProfile skills;
    Crease_End   playing_from_crease;
    
    BatStats bat_stats;
    BowlStats bowl_stats;
    
    bool is_preempted = false; 
    bool is_death_bowler = false; // Tag for Priority Scheduling
    double expected_stay_duration = 0.0;
} Player_stats;

typedef struct {
    BallType   type;
    double     actual_line, actual_length, actual_speed, difficulty;
    double     bat_control;         
    HitOutcome intended_outcome;    
    HitOutcome final_outcome;       
    int        runs_scored;
    
    std::string   fielder_name;        
    Player_stats* active_fielder;
    int           attempted_runs;
    double        expected_throw_time;
    double        actual_run_time;
    Crease_End    target_crease;
    bool          is_run_out;
    bool          comm_error; 
} Ball_state;

typedef struct {
    Ball_state      ball;
    Crease_End      bowling_crease;
    std::string     bowler_name;   
    pthread_mutex_t pitch_mutex;
} Pitch_state;

// ============================================================
// SCOREBOARD GLOBALS
// ============================================================
typedef struct {
    int total_runs;
    int wickets;
    int current_over;   
    int current_ball;   
    int fours;
    int sixes;
    int extras;
} ScoreBoard;

ScoreBoard      score;
pthread_mutex_t score_mutex;
pthread_mutex_t print_mutex;

std::atomic<int> target_score{-1}; 
std::atomic<int> current_innings{1};

int inn1_runs = 0, inn1_wkts = 0, inn1_overs = 0, inn1_balls = 0, inn1_extras = 0;
int inn2_runs = 0, inn2_wkts = 0, inn2_overs = 0, inn2_balls = 0, inn2_extras = 0;
bool india_bats_first = true;

const int MAX_OVERS = 20;

static void advance_ball_count() {
    score.current_ball++;
    if (score.current_ball == 6) {
        score.current_ball = 0;
        score.current_over++;
    }
}

static std::string score_line() {
    return std::to_string(score.total_runs) + "/"
         + std::to_string(score.wickets);
}

static std::string over_dot_ball() {
    return std::to_string(score.current_over) + "."
         + std::to_string(score.current_ball);
}

// ============================================================
// MATCH CONDITION PRIORITY ENCODING
// ============================================================
enum class InningsStatus { ONGOING = 0, TARGET_REACHED = 1, ALL_OUT = 2, OVERS_COMPLETED = 3 };

InningsStatus check_innings_status() {
    if (current_innings.load() == 2 && target_score.load() != -1 && score.total_runs >= target_score.load()) {
        return InningsStatus::TARGET_REACHED;
    }
    if (score.wickets >= 10) {
        return InningsStatus::ALL_OUT;
    }
    if (score.current_over >= MAX_OVERS) {
        return InningsStatus::OVERS_COMPLETED;
    }
    return InningsStatus::ONGOING;
}

// ============================================================
// COMMENTARY & GANTT HELPERS 
// ============================================================
static const int BOX_W = 120;

static std::string pad(const std::string& s, int width) {
    if ((int)s.size() >= width) return s;
    return s + std::string(width - s.size(), ' ');
}

static std::string truncate_pad(const std::string& s, int width) {
    std::string str = s;
    if ((int)str.size() > width - 1) {
        str = str.substr(0, width - 4) + "...";
    }
    return pad(str, width);
}

static void print_match_header(const std::string& striker,
                                const std::string& non_striker,
                                const std::string& bowler,
                                const std::string& batting_team_name,
                                const std::string& bowling_team_name) {
    pthread_mutex_lock(&print_mutex);
    std::ostringstream oss;
    oss << "\n";
    oss << "  +" << std::string(BOX_W, '=') << "+\n";
    oss << "  |" << pad("  T20 CRICKET SIMULATOR", BOX_W) << "|\n";
    oss << "  |" << pad("  " + batting_team_name + " v " + bowling_team_name + "  —  Innings " + std::to_string(current_innings.load()), BOX_W) << "|\n";
    oss << "  +" << std::string(BOX_W, '-') << "+\n";
    oss << "  |" << pad("  Striker     : " + striker,     BOX_W) << "|\n";
    oss << "  |" << pad("  Non-striker : " + non_striker, BOX_W) << "|\n";
    oss << "  |" << pad("  Bowler      : " + bowler,      BOX_W) << "|\n";
    oss << "  +" << std::string(BOX_W, '=') << "+\n";
    oss << "\n";
    std::string out = oss.str();
    comm_log << out;
    std::cout << out;
    pthread_mutex_unlock(&print_mutex);
}

static void print_over_header(const std::string& bowler, int over_num) {
    pthread_mutex_lock(&print_mutex);
    std::ostringstream oss;
    oss << "\n";
    oss << "  " << std::string(BOX_W, '-') << "\n";
    oss << "  OVER " << (over_num + 1) << "  —  " << bowler << " bowling";
    if (current_innings.load() == 2) {
        oss << "  |  Target: " << target_score.load() << "  |  Need " << (target_score.load() - score.total_runs) << " off " << (120 - (score.current_over * 6 + score.current_ball));
    }
    oss << "\n";
    oss << "  " << std::string(BOX_W, '-') << "\n";
    oss << "  "
        << std::left  << std::setw(8)  << "Over"
        << std::left  << std::setw(20) << "Bowler"
        << std::left  << std::setw(20) << "Batsman"
        << std::left  << std::setw(60) << "Commentary"
        << std::left  << std::setw(12) << "Score"
        << "\n";
    oss << "  " << std::string(BOX_W, '-') << "\n";
    std::string out = oss.str();
    comm_log << out;
    std::cout << out;
    pthread_mutex_unlock(&print_mutex);
}

static void print_delivery(const std::string& over_ball,
                            const std::string& bowler,
                            const std::string& batsman,
                            const std::string& commentary,
                            const std::string& scoreline) {
    pthread_mutex_lock(&print_mutex);
    std::ostringstream oss;
    oss << "  "
        << std::left << std::setw(8)  << truncate_pad(over_ball, 8)
        << std::left << std::setw(20) << truncate_pad(bowler, 20)
        << std::left << std::setw(20) << truncate_pad(batsman, 20)
        << std::left << std::setw(60) << truncate_pad(commentary, 60)
        << std::left << std::setw(12) << scoreline
        << "\n";
    std::string out = oss.str();
    comm_log << out;
    std::cout << out;
    pthread_mutex_unlock(&print_mutex);
}

static void print_end_of_over(const std::string& striker,
                               const std::string& non_striker) {
    pthread_mutex_lock(&print_mutex);
    std::ostringstream oss;
    oss << "\n";
    oss << "  " << std::string(BOX_W, '-') << "\n";
    oss << "  END OF OVER  —  "
        << score.total_runs << " runs  |  "
        << score.wickets    << " wickets  |  "
        << score.extras     << " extras\n";
    oss << "  At crease : " << striker << "  &  " << non_striker << "\n";
    oss << "  " << std::string(BOX_W, '-') << "\n";
    std::string out = oss.str();
    comm_log << out;
    std::cout << out;
    pthread_mutex_unlock(&print_mutex);
}

// --- FULL DETAILED GANTT PRINTER ---
void print_gantt_chart() {
    std::ostringstream oss;
    oss << "\n\n";
    oss << "==========================================================================================================\n";
    oss << "                                  OS THREAD EXECUTION GANTT CHART (TABULAR)                               \n";
    oss << "==========================================================================================================\n";
    oss << " " << std::left << std::setw(10) << "START(s)" 
        << "| " << std::left << std::setw(10) << "END(s)" 
        << "| " << std::left << std::setw(10) << "DURATION" 
        << "| " << std::left << std::setw(12) << "THREAD ROLE"
        << "| " << std::left << std::setw(22) << "PLAYER / ENTITY" 
        << "| " << std::left << std::setw(30) << "ACTIVITY" << "\n";
    oss << "----------------------------------------------------------------------------------------------------------\n";

    std::sort(gantt_log.begin(), gantt_log.end(), [](const GanttRecord& a, const GanttRecord& b) {
        return a.start_time < b.start_time;
    });

    for (const auto& record : gantt_log) {
        oss << " " << std::fixed << std::setprecision(4) << std::setw(9) << record.start_time 
            << " | " << std::fixed << std::setprecision(4) << std::setw(9) << record.end_time 
            << " | " << std::fixed << std::setprecision(4) << std::setw(9) << (record.end_time - record.start_time)
            << " | " << std::left << std::setw(11) << truncate_pad(record.thread_type, 11)
            << " | " << std::left << std::setw(21) << truncate_pad(record.player_name, 21)
            << " | " << std::left << truncate_pad(record.action, 40) << "\n";
    }
    oss << "==========================================================================================================\n\n";
    std::string out = oss.str();
    gantt_file << out; // Only outputs to the log file now!
}

void print_innings_scorecard(const std::string& team_name, const std::vector<Player_stats>& batting_team_vec, const std::vector<Player_stats>& bowling_team_vec, int total, int wkts, int ovs, int bls, int extras, std::ostringstream& oss) {
    oss << "  " << std::left << std::setw(71) << (team_name + " INNINGS")
        << std::right << std::setw(5) << "R" 
        << std::right << std::setw(5) << "B" 
        << std::right << std::setw(5) << "4s" 
        << std::right << std::setw(5) << "6s" 
        << std::right << std::setw(8) << "SR" << "\n";
    oss << "  " << std::string(BOX_W, '-') << "\n";
    for (const auto& p : batting_team_vec) {
        if (!p.bat_stats.has_batted) continue;
        double sr = (p.bat_stats.balls > 0) ? (p.bat_stats.runs * 100.0 / p.bat_stats.balls) : 0.0;
        oss << "  " << std::left << std::setw(20) << p.player_name 
            << std::left << std::setw(51) << p.bat_stats.dismissal
            << std::right << std::setw(5) << p.bat_stats.runs
            << std::right << std::setw(5) << p.bat_stats.balls
            << std::right << std::setw(5) << p.bat_stats.fours
            << std::right << std::setw(5) << p.bat_stats.sixes
            << std::right << std::setw(8) << std::fixed << std::setprecision(2) << sr << "\n";
    }
    oss << "  " << std::left << std::setw(20) << "Extras" 
        << std::left << std::setw(51) << ""
        << std::right << std::setw(5) << extras << "\n";
    oss << "  " << std::left << std::setw(20) << "Total" 
        << std::left << std::setw(51) << ("(" + std::to_string(wkts) + " wkts, " + std::to_string(ovs) + "." + std::to_string(bls) + " overs)")
        << std::right << std::setw(5) << total << "\n";
    oss << "  " << std::string(BOX_W, '-') << "\n";
    oss << "  " << std::left << std::setw(20) << "Bowling" 
        << std::right << std::setw(10) << "O" 
        << std::right << std::setw(5) << "M" 
        << std::right << std::setw(5) << "R" 
        << std::right << std::setw(5) << "W" 
        << std::right << std::setw(8) << "ECON" << "\n";
    for (const auto& p : bowling_team_vec) {
        if (p.bowl_stats.balls == 0 && p.bowl_stats.wides == 0 && p.bowl_stats.no_balls == 0) continue;
        int o = p.bowl_stats.balls / 6;
        int b = p.bowl_stats.balls % 6;
        double overs = o + (b / 6.0); 
        double econ = (overs > 0) ? (p.bowl_stats.runs / overs) : 0.0;
        std::string o_str = std::to_string(o) + "." + std::to_string(b);
        oss << "  " << std::left << std::setw(20) << p.player_name 
            << std::right << std::setw(10) << o_str
            << std::right << std::setw(5) << "0" 
            << std::right << std::setw(5) << p.bowl_stats.runs
            << std::right << std::setw(5) << p.bowl_stats.wickets
            << std::right << std::setw(8) << std::fixed << std::setprecision(2) << econ << "\n";
    }
    oss << "  " << std::string(BOX_W, '-') << "\n";
}

void print_final_scorecard() {
    pthread_mutex_lock(&print_mutex);
    std::ostringstream oss;
    oss << "\n\n";
    oss << "  ====================================================================================================\n";
    oss << "                                            FINAL SCORECARD                                           \n";
    oss << "  ====================================================================================================\n\n";
    
    extern std::vector<Player_stats> team_india;
    extern std::vector<Player_stats> team_aus;

    if (india_bats_first) {
        print_innings_scorecard("INDIA", team_india, team_aus, inn1_runs, inn1_wkts, inn1_overs, inn1_balls, inn1_extras, oss);
        oss << "\n";
        print_innings_scorecard("AUSTRALIA", team_aus, team_india, inn2_runs, inn2_wkts, inn2_overs, inn2_balls, inn2_extras, oss);
    } else {
        print_innings_scorecard("AUSTRALIA", team_aus, team_india, inn1_runs, inn1_wkts, inn1_overs, inn1_balls, inn1_extras, oss);
        oss << "\n";
        print_innings_scorecard("INDIA", team_india, team_aus, inn2_runs, inn2_wkts, inn2_overs, inn2_balls, inn2_extras, oss);
    }
    
    oss << "  ====================================================================================================\n\n";
    std::string out = oss.str();
    comm_log << out;
    std::cout << out;
    pthread_mutex_unlock(&print_mutex);
}

// ============================================================
// GLOBAL SYNCHRONISATION
// ============================================================
Pitch_state pitch;
sem_t       pitch_empty;       
sem_t       ball_ready_end1;   
sem_t       ball_ready_end2;   
sem_t       crease_capacity;   
sem_t       crease_swap_done;  

pthread_cond_t  cond_fielder_wake = PTHREAD_COND_INITIALIZER; 
pthread_mutex_t mutex_fielder_wake = PTHREAD_MUTEX_INITIALIZER; 
bool            ball_hit_to_field = false; 

sem_t       fielder_identified; 
sem_t       race_started;       
sem_t       fielder_done;      
pthread_mutex_t crease_mutex[2] = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER}; 

sem_t       non_striker_done; 

static std::atomic<int>  batsmen_ready{0};

// Persistent Data Storage
std::vector<Player_stats> team_india;
std::vector<Player_stats> team_aus;

// Active Pointers
std::vector<Player_stats>* batting_team = nullptr;
std::vector<Player_stats>* bowling_team = nullptr;

int next_bowler_idx = 6; 

Player_stats* active_bowler = nullptr;
Player_stats* active_batsman_end1 = nullptr;
Player_stats* active_batsman_end2 = nullptr;

pthread_t bowler_t;
pthread_t batsman_end1_t;
pthread_t batsman_end2_t;
pthread_t umpire_t;
std::vector<pthread_t> fielder_threads; 

sem_t umpire_sem;
std::atomic<bool> umpire_needs_bowler{false};
std::atomic<bool> wicket_fell{false};
std::atomic<Crease_End> umpire_needs_batsman_at;
std::atomic<int> victim_thread_slot{0}; 
std::atomic<bool> match_active{true};

// ============================================================
// BATTER SCHEDULING (FCFS vs SJF)
// ============================================================
Player_stats* find_next_batsman() {
    Player_stats* best_pick = nullptr;
    if (batter_scheduler_type == "sjf") {
        double min_burst = 999999.0;
        for (int i = 0; i < 11; i++) {
            if (!(*batting_team)[i].bat_stats.has_batted) {
                if ((*batting_team)[i].expected_stay_duration < min_burst) {
                    min_burst = (*batting_team)[i].expected_stay_duration;
                    best_pick = &(*batting_team)[i];
                }
            }
        }
    } else { // fcfs
        for (int i = 0; i < 11; i++) {
            if (!(*batting_team)[i].bat_stats.has_batted) {
                best_pick = &(*batting_team)[i];
                break;
            }
        }
    }
    
    if (best_pick) {
        best_pick->bat_stats.has_batted = true;
        best_pick->bat_stats.wait_time = get_sim_time() - best_pick->bat_stats.arrival_time;
    }
    return best_pick;
}

// ============================================================
// POPULATE PREDEFINED TEAMS
// ============================================================
void init_all_teams() {
    team_india.clear(); team_aus.clear();

    SkillProfile default_bat  = { 0.80, 0.80, 5.0, 0.0, 0.0, 0.80, 0.80, 0.80 };
    SkillProfile default_bowl = { 0.25, 0.20, 3.8, 0.9, 0.9, 0.80, 0.80, 0.80 };
    
    std::string ind_names[] = {"Virat Kohli", "Rohit Sharma", "Shubman Gill", "Suryakumar Yadav", "Rishabh Pant", "Hardik Pandya", "Ravindra Jadeja", "Axar Patel", "Kuldeep Yadav", "Jasprit Bumrah", "Mohammed Siraj"};
    for(int i = 0; i < 11; i++) {
        Player_Role r = (i < 6) ? Player_Role::BATSMAN : Player_Role::BOWLER; 
        team_india.push_back({ r, ind_names[i], "India", i+1, default_bat, Crease_End::END_1 });
    }
    team_india[0].skills = { 0.88, 0.92, 6.5, 0.00, 0.00, 0.90, 0.88, 0.85 };
    team_india[1].skills = { 0.85, 0.96, 5.8, 0.00, 0.00, 0.82, 0.80, 0.80 };
    team_india[2].skills = { 0.82, 0.92, 5.7, 0.90, 0.80, 0.85, 0.80, 0.82 };
    team_india[3].skills = { 0.90, 0.95, 6.3, 0.68, 0.85, 0.95, 0.92, 0.90 };
    team_india[4].skills = { 0.78, 0.94, 5.8, 0.92, 0.82, 0.88, 0.85, 0.85 };
    team_india[5].skills = { 0.75, 0.85, 6.0, 0.00, 0.00, 0.92, 0.88, 0.85 };
    team_india[6].skills = { 0.78, 0.88, 6.2, 0.96, 0.82, 0.85, 0.85, 0.85 };
    team_india[7].skills = { 0.60, 0.75, 5.5, 1.02, 0.94, 0.85, 0.85, 0.85 };
    team_india[8].skills = { 0.40, 0.60, 5.2, 1.06, 0.88, 0.75, 0.75, 0.80 };
    team_india[9].skills = { 0.30, 0.30, 5.0, 0.65, 0.95, 0.75, 0.75, 0.70 };
    team_india[10].skills = { 0.20, 0.30, 4.8, 0.98, 0.98, 0.80, 0.70, 0.75 };

    std::string aus_names[] = {"David Warner", "Travis Head", "Mitchell Marsh", "Glenn Maxwell", "Marcus Stoinis", "Matthew Wade", "Cameron Green", "Pat Cummins", "Mitchell Starc", "Adam Zampa", "Josh Hazlewood"};
    for(int i = 0; i < 11; i++) {
        Player_Role r = (i < 6) ? Player_Role::BATSMAN : Player_Role::BOWLER; 
        team_aus.push_back({ r, aus_names[i], "Australia", 100+i, default_bat, Crease_End::END_1 });
    }
    team_aus[0].skills = { 0.88, 0.92, 6.5, 0.00, 0.00, 0.90, 0.88, 0.85 };
    team_aus[1].skills = { 0.85, 0.96, 5.8, 0.00, 0.00, 0.82, 0.80, 0.80 };
    team_aus[2].skills = { 0.82, 0.92, 5.7, 0.90, 0.80, 0.85, 0.80, 0.82 };
    team_aus[3].skills = { 0.90, 0.95, 6.3, 0.68, 0.85, 0.95, 0.92, 0.90 };
    team_aus[4].skills = { 0.78, 0.94, 5.8, 0.92, 0.82, 0.88, 0.85, 0.85 };
    team_aus[5].skills = { 0.75, 0.85, 6.0, 0.00, 0.00, 0.92, 0.88, 0.85 };
    team_aus[6].skills = { 0.78, 0.88, 6.2, 0.96, 0.82, 0.85, 0.85, 0.85 };
    team_aus[7].skills = { 0.60, 0.75, 5.5, 1.02, 0.94, 0.85, 0.85, 0.85 };
    team_aus[8].skills = { 0.40, 0.60, 5.2, 1.06, 0.88, 0.75, 0.75, 0.80 };
    team_aus[9].skills = { 0.30, 0.30, 5.0, 0.65, 0.95, 0.75, 0.75, 0.70 };
    team_aus[10].skills = { 0.20, 0.30, 4.8, 0.98, 0.98, 0.80, 0.70, 0.75 };

    // --- ASSIGN DEATH OVER SPECIALISTS ---
    // India: Jasprit Bumrah (idx 9) and Mohammed Siraj (idx 10)
    team_india[9].is_death_bowler = true;
    team_india[10].is_death_bowler = true;

    // Australia: Mitchell Starc (idx 8) and Josh Hazlewood (idx 10)
    team_aus[8].is_death_bowler = true;
    team_aus[10].is_death_bowler = true;

    // --- ASSIGN EXPECTED BURST TIMES (FOR SJF) ---
    for (int i = 0; i < 11; i++) {
        team_india[i].expected_stay_duration = team_india[i].skills.technique * 100.0;
        team_aus[i].expected_stay_duration = team_aus[i].skills.technique * 100.0;
    }
}

// ============================================================
// SETUP INNINGS
// ============================================================
void setup_innings(int innings) {
    if (innings == 1) {
        batting_team = india_bats_first ? &team_india : &team_aus;
        bowling_team = india_bats_first ? &team_aus : &team_india;
    } else {
        batting_team = india_bats_first ? &team_aus : &team_india;
        bowling_team = india_bats_first ? &team_india : &team_aus;
    }

    for (int i=0; i<11; i++) {
        (*batting_team)[i].playing_from_crease = Crease_End::END_1; 
        (*bowling_team)[i].playing_from_crease = Crease_End::END_2;
        
        // Queue tracking for Wait Times
        (*batting_team)[i].bat_stats.has_batted = false;
        (*batting_team)[i].bat_stats.arrival_time = get_sim_time();
    }

    score = {0, 0, 0, 0, 0, 0, 0};
    next_bowler_idx = 6; 
    active_bowler = nullptr;
    active_batsman_end1 = nullptr;
    active_batsman_end2 = nullptr;
    batsmen_ready.store(0);
    umpire_needs_bowler.store(false);
    wicket_fell.store(false);
    match_active.store(true);
    victim_thread_slot.store(0);
    
    alloc_graph[0].store(-1); alloc_graph[1].store(-1);
    req_graph[0].store(-1); req_graph[1].store(-1);
    
    while(sem_trywait(&pitch_empty) == 0);
    sem_post(&pitch_empty); 
    while(sem_trywait(&ball_ready_end1) == 0);
    while(sem_trywait(&ball_ready_end2) == 0);
    while(sem_trywait(&crease_capacity) == 0);
    sem_post(&crease_capacity); sem_post(&crease_capacity); 
    while(sem_trywait(&crease_swap_done) == 0);
    
    pthread_mutex_lock(&mutex_fielder_wake);
    ball_hit_to_field = false;
    pthread_mutex_unlock(&mutex_fielder_wake);
    
    while(sem_trywait(&fielder_identified) == 0);
    while(sem_trywait(&race_started) == 0);
    while(sem_trywait(&fielder_done) == 0);
    while(sem_trywait(&umpire_sem) == 0);
    while(sem_trywait(&non_striker_done) == 0);
}

// ============================================================
// BOWLER THREAD
// ============================================================
void* bowler_thread(void* arg) {
    Player_stats* me = (Player_stats*)arg;
  
    pthread_mutex_lock(&score_mutex);
    int over_num = score.current_over;
    pthread_mutex_unlock(&score_mutex);

    print_over_header(me->player_name, over_num);

    int valid_balls_bowled = 0;

    while (valid_balls_bowled < 6 && match_active.load()) {

        if (check_innings_status() == InningsStatus::TARGET_REACHED) {
            match_active.store(false);
            break;
        }

        double t_wait = get_sim_time();
        sem_wait(&pitch_empty);
        if (!match_active.load()) break; 

        double t_start = get_sim_time(); 
        if (valid_balls_bowled > 0) log_gantt("Bowler", me->player_name, t_wait, t_start, "Waiting for Pitch (Blocked)");

        pthread_mutex_lock(&pitch.pitch_mutex);
        pitch.bowler_name = me->player_name;
        pitch.bowling_crease = me->playing_from_crease;

        // ── STAGE 1: BOWLER PIPELINE ──────────────────────────────────
        double intended_line   = threadRng.uniform(-1.0, 1.0);
        double intended_length = threadRng.uniform( 0.0, 1.0);
        double sigma           = 1.0 - me->skills.accuracy;

        pitch.ball.actual_line   = intended_line   + threadRng.gaussian(0.0, sigma);
        pitch.ball.actual_length = intended_length + threadRng.gaussian(0.0, sigma);
        pitch.ball.actual_speed  = me->skills.pace * 140.0
                                 + threadRng.gaussian(0.0, 5.0);

        double d_len  = 1.0 - std::abs(pitch.ball.actual_length - 0.5) * 2.0;
        double d_line = 1.0 - std::abs(pitch.ball.actual_line);
        double d_spd  = std::clamp((pitch.ball.actual_speed - 110.0) / 50.0, 0.0, 1.0);

        pitch.ball.difficulty = std::clamp(0.4 * d_len + 0.35 * d_spd + 0.25 * d_line, 0.0, 1.0);

        if (std::abs(pitch.ball.actual_line) > 1.3) {
            pitch.ball.type = BallType::WIDE;
            pthread_mutex_lock(&score_mutex);
            score.total_runs++; score.extras++;
            me->bowl_stats.runs++; me->bowl_stats.wides++;
            std::string ob = over_dot_ball(); std::string sl = score_line();
            pthread_mutex_unlock(&score_mutex);
            print_delivery(ob, me->player_name, "-", "Wide — down leg side  +1", sl);
        } else if (pitch.ball.actual_length > 0.95) {
            pitch.ball.type = BallType::NO_BALL;
            pthread_mutex_lock(&score_mutex);
            score.total_runs++; score.extras++;
            me->bowl_stats.runs++; me->bowl_stats.no_balls++;
            std::string ob = over_dot_ball(); std::string sl = score_line();
            pthread_mutex_unlock(&score_mutex);
            print_delivery(ob, me->player_name, "-", "No ball — front foot  +1", sl);
        } else {
            pitch.ball.type = BallType::NORMAL;
            valid_balls_bowled++;
            pthread_mutex_lock(&score_mutex);
            me->bowl_stats.balls++;
            pthread_mutex_unlock(&score_mutex);
        }

        pthread_mutex_unlock(&pitch.pitch_mutex);

        if (pitch.bowling_crease == Crease_End::END_1) sem_post(&ball_ready_end2);
        else sem_post(&ball_ready_end1);

        double t_end = get_sim_time(); 
        log_gantt("Bowler", me->player_name, t_start, t_end, "Delivers Ball");

        if (check_innings_status() == InningsStatus::TARGET_REACHED) {
            match_active.store(false);
            break;
        }
    }

    if (match_active.load()) {
        sem_wait(&pitch_empty); 
        sem_post(&pitch_empty); 
    }

    umpire_needs_bowler = true;
    sem_post(&umpire_sem);

    return NULL;
}

// ============================================================
// BATSMAN THREAD
// ============================================================
void* batsman_thread(void* arg) {
    Player_stats* me = (Player_stats*)arg;

    sem_wait(&crease_capacity);
    batsmen_ready++;   

    bool dismissed = false;

    while (!dismissed && match_active.load()) {

        double t_wait = get_sim_time();
        if (me->playing_from_crease == Crease_End::END_1) sem_wait(&ball_ready_end1);
        else sem_wait(&ball_ready_end2);

        if (!match_active.load()) break;

        double t_start = get_sim_time();
        log_gantt("Batsman", me->player_name, t_wait, t_start, "Waiting for Delivery (Blocked)");

        pthread_mutex_lock(&pitch.pitch_mutex);

        if (pitch.ball.type == BallType::RUNNING_PHASE) {
            pthread_mutex_unlock(&pitch.pitch_mutex);
            
            int home_crease = (int)me->playing_from_crease;
            int target_crease = (home_crease == 0) ? 1 : 0; 
            bool comm_error = pitch.ball.comm_error;
            
            double race_start_t = get_sim_time();
            bool safe = false;
            me->is_preempted = false;

            while (match_active.load() && !safe) {
                // 1. Hold Phase (Only if there is a mix-up)
                if (comm_error) {
                    pthread_mutex_lock(&crease_mutex[home_crease]);
                    alloc_graph[home_crease].store(me->player_id);
                }

                // 2. The Vulnerability Window (Organic Interleaving)
                usleep(threadRng.uniform(1000.0, 5000.0)); 

                usleep(pitch.ball.actual_run_time * pitch.ball.attempted_runs * TIME_SCALE * 1000000);

                // 3. The Wait Phase
                if (comm_error) {
                    req_graph[target_crease].store(me->player_id);
                }

                bool inner_preempted = false;

                // 4. Polling & Race Loop
                while (match_active.load()) {
                    if (pthread_mutex_trylock(&crease_mutex[target_crease]) == 0) {
                        safe = true;
                        if (comm_error) {
                            alloc_graph[target_crease].store(me->player_id);
                            req_graph[target_crease].store(-1);
                        }
                        break;
                    }
                    
                    if (me->is_preempted) {
                        if (comm_error) {
                            req_graph[target_crease].store(-1);
                            alloc_graph[home_crease].store(-1);
                            pthread_mutex_unlock(&crease_mutex[home_crease]);
                        }
                        
                        log_gantt("Batsman", me->player_name, get_sim_time(), get_sim_time()+0.001, "Yields Crease (Preempted)");
                        
                        pthread_mutex_lock(&print_mutex);
                        std::string out = "  [DEADLOCK RESOLVED] " + me->player_name + " yields and turns back to break the circular wait!\n";
                        comm_log << out;
                        std::cout << out;
                        pthread_mutex_unlock(&print_mutex);
                        
                        usleep(300000 * TIME_SCALE); // Penalty wait time
                        me->is_preempted = false;
                        comm_error = false; 
                        inner_preempted = true; 
                        break; // Break inner polling to restart Hold Phase cleanly
                    }

                    if (pitch.ball.is_run_out && pitch.ball.target_crease == (Crease_End)target_crease) {
                        safe = false;
                        break;
                    }
                    usleep(1000);
                }

                // 5. Retry Mechanism
                if (inner_preempted) {
                    continue; 
                }

                // 6. Cleanup Phase (Successful or Run Out)
                if (comm_error) {
                    alloc_graph[home_crease].store(-1);
                    pthread_mutex_unlock(&crease_mutex[home_crease]);
                }
                break;
            }
            if (safe) {
                 if (comm_error) alloc_graph[target_crease].store(-1);
                 pthread_mutex_unlock(&crease_mutex[target_crease]);
            }
            
            sem_post(&non_striker_done);
            continue; 
        }

        if (pitch.ball.type != BallType::NORMAL) {
            if (pitch.ball.type == BallType::CREASE_SWAP) {
                Player_stats* temp = active_batsman_end1;
                active_batsman_end1 = active_batsman_end2;
                active_batsman_end2 = temp;
                
                if (active_batsman_end1) active_batsman_end1->playing_from_crease = Crease_End::END_1;
                if (active_batsman_end2) active_batsman_end2->playing_from_crease = Crease_End::END_2;
                
                pthread_t temp_t = batsman_end1_t; batsman_end1_t = batsman_end2_t; batsman_end2_t = temp_t;
                pthread_mutex_unlock(&pitch.pitch_mutex);
                
                usleep(15000 * TIME_SCALE); 
                double t_end = get_sim_time();
                log_gantt("Batsman", me->player_name, t_start, t_end, "Non-Striker crosses crease");

                sem_post(&crease_swap_done); 
                continue; 
            } else {
                pthread_mutex_unlock(&pitch.pitch_mutex);
                usleep(5000 * TIME_SCALE); 
                double t_end = get_sim_time();
                log_gantt("Batsman", me->player_name, t_start, t_end, "Leaves Extra Delivery");
                sem_post(&pitch_empty);
                continue;
            }
        }

        double bat_ctrl = me->skills.technique - pitch.ball.difficulty + threadRng.gaussian(0.0, 0.15);

        double p_miss   = sigmoid(-bat_ctrl * 3.0) * 0.15; 
        double lbw_f    = std::exp(-std::pow(pitch.ball.actual_line, 2) / 0.2) * std::exp(-std::pow(pitch.ball.actual_length - 0.5, 2) / 0.3);
        double p_lbw    = sigmoid(-bat_ctrl * 2.0) * (1.0 - p_miss) * lbw_f * 0.15; 
        double edge_f   = 1.0 - sigmoid(bat_ctrl * 3.0); 
        double p_caught = sigmoid(-bat_ctrl * 1.5) * (1.0 - p_miss - p_lbw) * edge_f * 0.25; 
        double p_six    = sigmoid(bat_ctrl * 2.0 + me->skills.power - 1.0) * (1.0 - p_miss - p_lbw - p_caught) * 0.10; 
        double p_four   = sigmoid(bat_ctrl * 1.5) * (1.0 - p_miss - p_lbw - p_caught - p_six) * 0.20; 
        double p_con    = std::max(0.0, 1.0 - p_miss - p_lbw - p_caught - p_six - p_four);

        std::vector<double> probs = {p_miss, p_lbw, p_caught, p_six, p_four, p_con};
        
        pitch.ball.bat_control = bat_ctrl;
        pitch.ball.intended_outcome = static_cast<HitOutcome>(threadRng.categorical(probs));
        pitch.ball.final_outcome = pitch.ball.intended_outcome; 
        pitch.ball.runs_scored = 0;
        pitch.ball.is_run_out = false;
        
        bool needs_fielder = (pitch.ball.intended_outcome == HitOutcome::FOUR ||
                              pitch.ball.intended_outcome == HitOutcome::CAUGHT ||
                              pitch.ball.intended_outcome == HitOutcome::CONNECT);

        usleep(10000 * TIME_SCALE); 

        if (needs_fielder) {
            pthread_mutex_lock(&mutex_fielder_wake);
            ball_hit_to_field = true;
            pthread_cond_signal(&cond_fielder_wake);
            pthread_mutex_unlock(&mutex_fielder_wake);
            
            pthread_mutex_unlock(&pitch.pitch_mutex);
            sem_wait(&fielder_identified);
            pthread_mutex_lock(&pitch.pitch_mutex);

            Player_stats* f = pitch.ball.active_fielder;
            double fielding_eff = (f->skills.agility + f->skills.throwing_accuracy) / 2.0;
            double expected_throw_time = 4.0 + 4.0 * (1.0 - fielding_eff);
            pitch.ball.expected_throw_time = expected_throw_time;
            
            Player_stats* non_striker = (me == active_batsman_end1) ? active_batsman_end2 : active_batsman_end1;
            double min_spd = std::min(me->skills.running_speed, non_striker->skills.running_speed);
            double expected_run_time = 18.0 / min_spd;

            int attempted_runs = 0;
            if (pitch.ball.intended_outcome == HitOutcome::CONNECT) {
                attempted_runs = std::floor(expected_throw_time / expected_run_time);
            }
            pitch.ball.attempted_runs = attempted_runs;

            if (attempted_runs > 0) {
                pitch.ball.actual_run_time = expected_run_time + std::max(0.0, threadRng.gaussian(0.0, 0.2));
                pitch.ball.target_crease = (attempted_runs % 2 != 0) ? 
                    ((me->playing_from_crease == Crease_End::END_1) ? Crease_End::END_2 : Crease_End::END_1) : 
                    me->playing_from_crease;
            }

            if (attempted_runs > 0 && pitch.ball.intended_outcome == HitOutcome::CONNECT) {
                pitch.ball.comm_error = (attempted_runs % 2 != 0) && (threadRng.uniform(0.0, 1.0) < 0.1); 
                pitch.ball.type = BallType::RUNNING_PHASE;
                
                if (me->playing_from_crease == Crease_End::END_1) sem_post(&ball_ready_end2);
                else sem_post(&ball_ready_end1);
            }

            sem_post(&race_started);
            pthread_mutex_unlock(&pitch.pitch_mutex);

            if (attempted_runs > 0 && pitch.ball.intended_outcome == HitOutcome::CONNECT) {
                
                int home_crease = (int)me->playing_from_crease;
                int target_crease = (int)pitch.ball.target_crease; 
                bool comm_error = pitch.ball.comm_error;

                double race_start_t = get_sim_time();
                bool safe = false;
                me->is_preempted = false;

                while (match_active.load() && !safe) {
                    // 1. Hold Phase
                    if (comm_error) {
                        pthread_mutex_lock(&crease_mutex[home_crease]);
                        alloc_graph[home_crease].store(me->player_id);
                    }

                    // 2. The Vulnerability Window (Organic Interleaving)
                    usleep(threadRng.uniform(1000.0, 5000.0)); 

                    usleep(pitch.ball.actual_run_time * attempted_runs * TIME_SCALE * 1000000);

                    // 3. The Wait Phase
                    if (comm_error) {
                        req_graph[target_crease].store(me->player_id);
                    }
                    
                    bool inner_preempted = false;

                    // 4. Polling & Race Loop
                    while (match_active.load()) {
                        if (pthread_mutex_trylock(&crease_mutex[target_crease]) == 0) {
                            safe = true;
                            if (comm_error) {
                                alloc_graph[target_crease].store(me->player_id);
                                req_graph[target_crease].store(-1);
                            }
                            break;
                        }
                        
                        if (me->is_preempted) {
                            if (comm_error) {
                                req_graph[target_crease].store(-1);
                                alloc_graph[home_crease].store(-1);
                                pthread_mutex_unlock(&crease_mutex[home_crease]);
                            }
                            
                            log_gantt("Batsman", me->player_name, get_sim_time(), get_sim_time()+0.001, "Yields Crease (Preempted)");
                            
                            pthread_mutex_lock(&print_mutex);
                            std::string out = "  [DEADLOCK RESOLVED] " + me->player_name + " yields and turns back to break the circular wait!\n";
                            comm_log << out;
                            std::cout << out;
                            pthread_mutex_unlock(&print_mutex);
                            
                            usleep(300000 * TIME_SCALE); // Penalty wait time
                            me->is_preempted = false;
                            comm_error = false; 
                            inner_preempted = true; 
                            break; // Break inner polling to restart Hold Phase cleanly
                        }

                        if (pitch.ball.is_run_out && pitch.ball.target_crease == (Crease_End)target_crease) {
                            safe = false;
                            break;
                        }
                        usleep(1000);
                    }

                    // 5. Retry Mechanism
                    if (inner_preempted) {
                        continue; 
                    }

                    // 6. Cleanup Phase
                    if (comm_error) {
                        alloc_graph[home_crease].store(-1);
                        pthread_mutex_unlock(&crease_mutex[home_crease]);
                    }
                    break;
                }

                double race_end_t = get_sim_time();
                log_gantt("Batsman", me->player_name, race_start_t, race_end_t, "Racing for Crease");

                sem_wait(&fielder_done);
                
                pthread_mutex_lock(&pitch.pitch_mutex);
                if (safe && !pitch.ball.is_run_out) {
                    pitch.ball.runs_scored = attempted_runs;
                } else if (!safe && pitch.ball.is_run_out) {
                    dismissed = true; 
                }
                
                // CRITICAL FIX: Unlock the target crease BEFORE waiting for the Non-Striker!
                if (safe) {
                    if (comm_error) alloc_graph[target_crease].store(-1);
                    pthread_mutex_unlock(&crease_mutex[target_crease]); 
                }
                pthread_mutex_unlock(&pitch.pitch_mutex); // Drop pitch lock while sleeping
                
                sem_wait(&non_striker_done); 
                
                pthread_mutex_lock(&pitch.pitch_mutex); // Re-acquire for Scoreboard phase
            } else {
                sem_wait(&fielder_done);
                pthread_mutex_lock(&pitch.pitch_mutex);
            }

        } else if (pitch.ball.intended_outcome == HitOutcome::SIX) {
            pitch.ball.runs_scored = 6;
        }

        std::string bowler = pitch.bowler_name;
        std::string commentary;
        bool needs_crease_swap = false;

        pthread_mutex_lock(&score_mutex);
        
        me->bat_stats.balls++;

        switch (pitch.ball.final_outcome) {
            case HitOutcome::SIX:
                score.total_runs += 6; score.sixes++; advance_ball_count();
                me->bat_stats.runs += 6; me->bat_stats.sixes++;
                active_bowler->bowl_stats.runs += 6;
                commentary = "SIX! Cleared the ropes!"; break;
            case HitOutcome::FOUR:
                score.total_runs += 4; score.fours++; advance_ball_count();
                me->bat_stats.runs += 4; me->bat_stats.fours++;
                active_bowler->bowl_stats.runs += 4;
                commentary = (pitch.ball.intended_outcome == HitOutcome::CONNECT) ? 
                    "FOUR! Misfield by " + pitch.ball.fielder_name + "!" : "FOUR! Couldn't cut it off."; 
                break;
            case HitOutcome::LBW:
                score.wickets++; advance_ball_count(); dismissed = true;
                active_bowler->bowl_stats.wickets++;
                commentary = "OUT! LBW — plumb in front!"; break;
            case HitOutcome::CAUGHT:
                score.wickets++; advance_ball_count(); dismissed = true;
                active_bowler->bowl_stats.wickets++;
                commentary = "OUT! Caught beautifully by " + pitch.ball.fielder_name + "!"; break;
            case HitOutcome::MISS:
                advance_ball_count();
                commentary = "Dot ball — beaten outside off."; break;
            case HitOutcome::CONNECT: {
                if (pitch.ball.is_run_out) {
                    score.total_runs += pitch.ball.runs_scored;
                    score.wickets++; advance_ball_count(); dismissed = true;
                    me->bat_stats.runs += pitch.ball.runs_scored;
                    active_bowler->bowl_stats.runs += pitch.ball.runs_scored;
                    commentary = "RUN OUT! " + me->player_name + " falls short trying for " + std::to_string(pitch.ball.attempted_runs) + ". Completed " + std::to_string(pitch.ball.runs_scored) + ".";
                } else {
                    int runs_ran = pitch.ball.runs_scored;
                    score.total_runs += runs_ran;
                    advance_ball_count();
                    me->bat_stats.runs += runs_ran;
                    active_bowler->bowl_stats.runs += runs_ran;
                    commentary = "Pushed to " + pitch.ball.fielder_name + ". They safely run " + std::to_string(runs_ran) + ".";
                    if (runs_ran % 2 != 0) needs_crease_swap = true; 
                }
                break;
            }
        }

        if (dismissed) {
            me->bat_stats.is_out = true;
            if (pitch.ball.final_outcome == HitOutcome::CAUGHT) {
                me->bat_stats.dismissal = "c " + pitch.ball.fielder_name + " b " + pitch.bowler_name;
            } else if (pitch.ball.final_outcome == HitOutcome::LBW) {
                me->bat_stats.dismissal = "lbw b " + pitch.bowler_name;
            } else if (pitch.ball.is_run_out) {
                me->bat_stats.dismissal = "run out (" + pitch.ball.fielder_name + ")";
            }
        }

        std::string ob = over_dot_ball();
        std::string sl = score_line();
        pthread_mutex_unlock(&score_mutex);

        print_delivery(ob, bowler, me->player_name, commentary, sl);

        if (dismissed) {
            pthread_mutex_lock(&print_mutex);
            std::string out = "\n  *** WICKET *** " + me->player_name + " is out.  " + sl + "\n\n";
            comm_log << out;
            std::cout << out;
            pthread_mutex_unlock(&print_mutex);
            
            int my_slot = (me == active_batsman_end1) ? 1 : 2;
            victim_thread_slot = my_slot;
            
            if (pitch.ball.is_run_out) {
                Crease_End run_out_crease = pitch.ball.target_crease;
                Crease_End opposite_crease = (run_out_crease == Crease_End::END_1) ? Crease_End::END_2 : Crease_End::END_1;
                
                if (my_slot == 1) active_batsman_end2->playing_from_crease = run_out_crease;
                else active_batsman_end1->playing_from_crease = run_out_crease;
                
                umpire_needs_batsman_at = opposite_crease;
            } else {
                umpire_needs_batsman_at = me->playing_from_crease;
            }

            wicket_fell = true;
            sem_post(&umpire_sem);
        }

        double t_end = get_sim_time();
        log_gantt("Batsman", me->player_name, t_start, t_end, "Processes Scoreboard / Dismissal");

        if (needs_crease_swap) {
            pitch.ball.type = BallType::CREASE_SWAP; 
            if (me->playing_from_crease == Crease_End::END_1) sem_post(&ball_ready_end2);
            else sem_post(&ball_ready_end1);
            
            pthread_mutex_unlock(&pitch.pitch_mutex);
            double t_wait_swap = get_sim_time();
            sem_wait(&crease_swap_done); 
            double t_swap_done = get_sim_time();
            log_gantt("Batsman", me->player_name, t_wait_swap, t_swap_done, "Waiting for Non-Striker Handshake");
        } else {
            pthread_mutex_unlock(&pitch.pitch_mutex);
        }

        sem_post(&pitch_empty);
    }

    sem_post(&crease_capacity);
    return NULL;
}

// ============================================================
// FIELDER THREAD
// ============================================================
void* fielder_thread(void* arg) {
    Player_stats* me = (Player_stats*)arg;

    while(match_active.load()) {
        pthread_mutex_lock(&mutex_fielder_wake);
        while (!ball_hit_to_field && match_active.load()) {
            pthread_cond_wait(&cond_fielder_wake, &mutex_fielder_wake);
        }
        if (!match_active.load()) {
            pthread_mutex_unlock(&mutex_fielder_wake);
            break;
        }
        ball_hit_to_field = false; 
        pthread_mutex_unlock(&mutex_fielder_wake);
        
        double t_start = get_sim_time();

        pthread_mutex_lock(&pitch.pitch_mutex);
        pitch.ball.active_fielder = me;
        pitch.ball.fielder_name = me->player_name;
        sem_post(&fielder_identified);
        pthread_mutex_unlock(&pitch.pitch_mutex);

        sem_wait(&race_started);
        pthread_mutex_lock(&pitch.pitch_mutex);

        if (pitch.ball.intended_outcome == HitOutcome::CONNECT) {
            if (pitch.ball.attempted_runs > 0) {
                double actual_throw = pitch.ball.expected_throw_time + threadRng.gaussian(0.0, 0.5);
                pthread_mutex_unlock(&pitch.pitch_mutex);
                
                usleep(actual_throw * TIME_SCALE * 1000000);
                
                if (pthread_mutex_trylock(&crease_mutex[(int)pitch.ball.target_crease]) == 0) {
                    pthread_mutex_lock(&pitch.pitch_mutex);
                    pitch.ball.is_run_out = true;
                    pitch.ball.runs_scored = std::floor(actual_throw / pitch.ball.actual_run_time);
                    pthread_mutex_unlock(&pitch.pitch_mutex);
                    pthread_mutex_unlock(&crease_mutex[(int)pitch.ball.target_crease]);
                }
                
                double t_end = get_sim_time();
                log_gantt("Fielder", me->player_name, t_start, t_end, "Throwing to Stumps (Race)");
                pthread_mutex_lock(&pitch.pitch_mutex);
            } else {
                pitch.ball.final_outcome = HitOutcome::CONNECT;
                pitch.ball.runs_scored = 0;
                double t_end = get_sim_time();
                log_gantt("Fielder", me->player_name, t_start, t_end, "Fields the ball (No run)");
            }
        }
        else if (pitch.ball.intended_outcome == HitOutcome::FOUR) {
            double save_prob = me->skills.agility * sigmoid(me->skills.catching_skill - 0.6) + threadRng.gaussian(0.0, 0.05);
            usleep(15000 * TIME_SCALE); 
            double t_end = get_sim_time();

            if (save_prob > 0.75) {
                pitch.ball.final_outcome = HitOutcome::CONNECT; pitch.ball.runs_scored = 2; 
                log_gantt("Fielder", me->player_name, t_start, t_end, "Saves Boundary!");
            } else {
                pitch.ball.final_outcome = HitOutcome::FOUR; pitch.ball.runs_scored = 4;
                log_gantt("Fielder", me->player_name, t_start, t_end, "Chases ball (Boundary)");
            }
        }
        else if (pitch.ball.intended_outcome == HitOutcome::CAUGHT) {
            double catch_diff = (1.0 - pitch.ball.bat_control)*0.5 + pitch.ball.difficulty*0.3 + threadRng.uniform(0.0, 1.0)*0.2;
            double catch_success = me->skills.catching_skill - catch_diff + threadRng.gaussian(0.0, 0.12);
            usleep(10000 * TIME_SCALE); 
            double t_end = get_sim_time();

            if (catch_success > 0) {
                pitch.ball.final_outcome = HitOutcome::CAUGHT; pitch.ball.runs_scored = 0;
                log_gantt("Fielder", me->player_name, t_start, t_end, "Takes the Catch!");
            } else {
                pitch.ball.final_outcome = HitOutcome::CONNECT; pitch.ball.runs_scored = 1; 
                log_gantt("Fielder", me->player_name, t_start, t_end, "Drops the Catch");
            }
        }

        pthread_mutex_unlock(&pitch.pitch_mutex);
        sem_post(&fielder_done);
    }
    return NULL;
}

// ============================================================
// UMPIRE THREAD 
// ============================================================
void* umpire_thread(void* arg) {

    active_batsman_end1 = find_next_batsman(); 
    if (active_batsman_end1) {
        active_batsman_end1->playing_from_crease = Crease_End::END_1;
        pthread_create(&batsman_end1_t, NULL, batsman_thread, active_batsman_end1);
    }
    
    active_batsman_end2 = find_next_batsman(); 
    if (active_batsman_end2) {
        active_batsman_end2->playing_from_crease = Crease_End::END_2;
        pthread_create(&batsman_end2_t, NULL, batsman_thread, active_batsman_end2);
    }

    while (batsmen_ready.load() < 2) {
        struct timespec ts = {0, 500000L};   
        nanosleep(&ts, nullptr);
    }

    active_bowler = &(*bowling_team)[next_bowler_idx];
    active_bowler->playing_from_crease = Crease_End::END_2;
    pthread_create(&bowler_t, NULL, bowler_thread, active_bowler);

    while (match_active.load()) {
        
        if (alloc_graph[0].load() != -1 && alloc_graph[1].load() != -1 && 
            req_graph[0].load() == alloc_graph[1].load() && req_graph[1].load() == alloc_graph[0].load()) {
            
            double t_st = get_sim_time();
            Player_stats* victim = active_batsman_end2; 
            victim->is_preempted = true;
            
            req_graph[0].store(-1);
            req_graph[1].store(-1);
            
            log_gantt("Umpire", "Deadlock Daemon", t_st, get_sim_time(), "Detects Deadlock, Preempts Batsman");

            pthread_mutex_lock(&print_mutex);
            std::string out = "\n  [UMPIRE OS DAEMON] DEADLOCK DETECTED! Mix-up in the middle between batsmen! Preempting " + victim->player_name + "...\n";
            comm_log << out;
            std::cout << out;
            pthread_mutex_unlock(&print_mutex);
        }

        if (sem_trywait(&umpire_sem) == 0) {
            double t_start = get_sim_time();
            
            if (wicket_fell.load()) {
                wicket_fell = false;
                
                if (victim_thread_slot.load() == 1) {
                    if (active_batsman_end1) pthread_join(batsman_end1_t, NULL); 
                    if (score.wickets < 10) {
                        active_batsman_end1 = find_next_batsman();
                        if (active_batsman_end1) {
                            active_batsman_end1->playing_from_crease = umpire_needs_batsman_at;
                            pthread_create(&batsman_end1_t, NULL, batsman_thread, active_batsman_end1);
                        }
                    } else {
                        active_batsman_end1 = nullptr; 
                    }
                } else {
                    if (active_batsman_end2) pthread_join(batsman_end2_t, NULL); 
                    if (score.wickets < 10) {
                        active_batsman_end2 = find_next_batsman();
                        if (active_batsman_end2) {
                            active_batsman_end2->playing_from_crease = umpire_needs_batsman_at;
                            pthread_create(&batsman_end2_t, NULL, batsman_thread, active_batsman_end2);
                        }
                    } else {
                        active_batsman_end2 = nullptr;
                    }
                }
                
                usleep(40000 * TIME_SCALE); 
                double t_end = get_sim_time();
                log_gantt("Umpire", "OS Scheduler", t_start, t_end, "Context Switch / Crease Setup");
            }
            if (umpire_needs_bowler.load()) {
                umpire_needs_bowler = false;
                if (active_bowler) pthread_join(bowler_t, NULL); 
                std::string str1 = active_batsman_end1 ? active_batsman_end1->player_name : "";
                std::string str2 = active_batsman_end2 ? active_batsman_end2->player_name : "";
                print_end_of_over(str1, str2);

                if (score.current_over < MAX_OVERS && score.wickets < 10) {
                    Crease_End new_crease = (active_bowler->playing_from_crease == Crease_End::END_1) ? Crease_End::END_2 : Crease_End::END_1;
                    
                    Player_stats* next_bowler = nullptr;
                    
                    // Pre-calculate the strict Round Robin bowler to know who gets replaced
                    Player_stats* strict_rr_bowler = nullptr;
                    int last_idx = 5; 
                    for (int i = 6; i < 11; i++) {
                        if (&(*bowling_team)[i] == active_bowler) last_idx = i;
                    }
                    for (int offset = 1; offset <= 5; offset++) {
                        int test_idx = 6 + ((last_idx - 6 + offset) % 5);
                        Player_stats* b = &(*bowling_team)[test_idx];
                        if (b != active_bowler && b->bowl_stats.balls < 24) {
                            strict_rr_bowler = b;
                            break;
                        }
                    }
                    
                    if (bowler_scheduler_type == "priority") {
                        bool high_intensity = false;
                        
                        // Condition 1: Last 2 overs of the match
                        if (score.current_over >= 18) high_intensity = true;
                        
                        // Condition 2: Defending a tight chase (Chasing team needs <= 10% of target)
                        if (current_innings.load() == 2 && target_score.load() > 0) {
                            int runs_needed = target_score.load() - score.total_runs;
                            if (runs_needed > 0 && runs_needed <= (0.10 * target_score.load())) {
                                high_intensity = true;
                            }
                        }
                        
                        std::vector<Player_stats*> normal_candidates;
                        std::vector<Player_stats*> death_candidates;
                        
                        // Filter available bowlers (Must not have bowled last over, Must have balls < 24)
                        for (int i = 6; i < 11; i++) {
                            Player_stats* b = &(*bowling_team)[i];
                            if (b != active_bowler && b->bowl_stats.balls < 24) {
                                if (b->is_death_bowler) death_candidates.push_back(b);
                                else normal_candidates.push_back(b);
                            }
                        }
                        
                        // Sort by freshness to balance workload inside the tiers
                        auto sort_freshness = [](Player_stats* a, Player_stats* b) {
                            return a->bowl_stats.balls < b->bowl_stats.balls;
                        };
                        std::sort(normal_candidates.begin(), normal_candidates.end(), sort_freshness);
                        std::sort(death_candidates.begin(), death_candidates.end(), sort_freshness);
                        
                        if (high_intensity && !death_candidates.empty()) {
                            next_bowler = death_candidates.front();
                            
                            pthread_mutex_lock(&print_mutex);
                            std::string out;
                            if (strict_rr_bowler && strict_rr_bowler != next_bowler) {
                                out = "  [SCHEDULER] High Intensity Phase! Deploying Death Over Specialist: " + next_bowler->player_name + " in place of " + strict_rr_bowler->player_name + "\n";
                            } else {
                                out = "  [SCHEDULER] High Intensity Phase! Deploying Death Over Specialist: " + next_bowler->player_name + "\n";
                            }
                            comm_log << out; std::cout << out;
                            pthread_mutex_unlock(&print_mutex);
                        } else if (!high_intensity && !normal_candidates.empty()) {
                            next_bowler = normal_candidates.front();
                        } else {
                            // Fallback: If normal bowlers reach 4-over quota early, death bowlers must bowl.
                            if (!death_candidates.empty()) next_bowler = death_candidates.front();
                            else if (!normal_candidates.empty()) next_bowler = normal_candidates.front();
                        }
                    } 
                    
                    // Fallback to strict Round Robin if priority is off (or as an ultimate failsafe)
                    if (next_bowler == nullptr) {
                        next_bowler = strict_rr_bowler;
                    }
                    
                    active_bowler = next_bowler;
                    active_bowler->playing_from_crease = new_crease;
                    pthread_create(&bowler_t, NULL, bowler_thread, active_bowler);
                } else {
                    active_bowler = nullptr;
                }
                
                usleep(40000 * TIME_SCALE); 
                double t_end = get_sim_time();
                log_gantt("Umpire", "OS Scheduler", t_start, t_end, "RR Context Switch (Over End)");
            }
        }
        
        if (check_innings_status() != InningsStatus::ONGOING) {
            match_active = false;
            break;
        }

        usleep(1000); 
    }

    match_active.store(false);
    sem_post(&ball_ready_end1);
    sem_post(&ball_ready_end2);
    sem_post(&pitch_empty); 
    
    pthread_mutex_lock(&mutex_fielder_wake);
    pthread_cond_broadcast(&cond_fielder_wake);
    pthread_mutex_unlock(&mutex_fielder_wake);

    if (active_bowler) pthread_join(bowler_t, NULL);
    if (active_batsman_end1) pthread_join(batsman_end1_t, NULL);
    if (active_batsman_end2) pthread_join(batsman_end2_t, NULL);

    return NULL;
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char* argv[]) {
    // --- COMMAND LINE PARSING ---
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-bat" && i + 1 < argc) {
            batter_scheduler_type = argv[++i];
        } else if (std::string(argv[i]) == "-bowl" && i + 1 < argc) {
            bowler_scheduler_type = argv[++i];
        }
    }
    
    // Basic validation
    if (batter_scheduler_type != "sjf") batter_scheduler_type = "fcfs"; // Fallback to FCFS
    if (bowler_scheduler_type != "priority") bowler_scheduler_type = "rr"; // Fallback to RR

    comm_log.open("commentary_log.txt");
    gantt_file.open("gantt_log.txt");
    wait_log.open("wait_times.csv", std::ios::app);

    if (!comm_log.is_open() || !gantt_file.is_open() || !wait_log.is_open()) {
        std::cerr << "Failed to open output log files!\n";
        return 1;
    }
    
    wait_log.seekp(0, std::ios::end);
    if (wait_log.tellp() == 0) {
        wait_log << "Scheduler,Innings,Player,Expected_Burst,Wait_Time\n";
    }

    std::cout << "Starting Match Simulator...\n";
    std::cout << "Batting Scheduler : " << batter_scheduler_type << "\n";
    std::cout << "Bowling Scheduler : " << bowler_scheduler_type << "\n";
    std::cout << "Writing Match Commentary to 'commentary_log.txt'...\n";
    std::cout << "Writing Full, Cluttered Thread Execution Gantt Chart to 'gantt_log.txt'...\n";

    sim_start_time = std::chrono::steady_clock::now(); 

    score = {0, 0, 0, 0, 0, 0, 0};
    pthread_mutex_init(&score_mutex, NULL);
    pthread_mutex_init(&print_mutex, NULL);
    pthread_mutex_init(&pitch.pitch_mutex, NULL);
    sem_init(&pitch_empty,     0, 1);  
    sem_init(&ball_ready_end1, 0, 0);
    sem_init(&ball_ready_end2, 0, 0);
    sem_init(&crease_capacity, 0, 2);  
    sem_init(&crease_swap_done, 0, 0); 
    
    pthread_cond_init(&cond_fielder_wake, NULL);
    pthread_mutex_init(&mutex_fielder_wake, NULL);

    sem_init(&fielder_identified, 0, 0);
    sem_init(&race_started,    0, 0);
    sem_init(&fielder_done,    0, 0);
    sem_init(&umpire_sem,      0, 0); 
    
    sem_init(&non_striker_done, 0, 0); 
    alloc_graph[0].store(-1); alloc_graph[1].store(-1);
    req_graph[0].store(-1); req_graph[1].store(-1);

    init_all_teams(); 

    india_bats_first = (threadRng.uniform(0.0, 1.0) > 0.5);
    
    std::string team1 = india_bats_first ? "India" : "Australia";
    std::string team2 = india_bats_first ? "Australia" : "India";

    pthread_mutex_lock(&print_mutex);
    std::ostringstream oss;
    oss << "\n\n  +" << std::string(BOX_W, '=') << "+\n";
    oss << "  |" << pad("  TOSS RESULT", BOX_W) << "|\n";
    oss << "  |" << pad(std::string("  Coin flipped... ") + team1 + " won the toss and elected to BAT first!", BOX_W) << "|\n";
    oss << "  +" << std::string(BOX_W, '=') << "+\n\n";
    std::string out = oss.str();
    comm_log << out;
    std::cout << out;
    pthread_mutex_unlock(&print_mutex);

    // ==========================================
    // INNINGS 1
    // ==========================================
    current_innings.store(1);
    setup_innings(1); 
    print_match_header((*batting_team)[0].player_name, (*batting_team)[1].player_name, (*bowling_team)[6].player_name, (*batting_team)[0].player_team, (*bowling_team)[0].player_team);

    fielder_threads.resize(11);
    for(int i = 0; i < 11; i++) {
        pthread_create(&fielder_threads[i], NULL, fielder_thread, &(*bowling_team)[i]);
    }

    pthread_create(&umpire_t, NULL, umpire_thread, NULL);
    pthread_join(umpire_t, NULL); 

    for(int i = 0; i < 11; i++) {
        pthread_join(fielder_threads[i], NULL); 
    }

    // --- LOG WAIT TIMES FOR INNINGS 1 ---
    for (const auto& player : *batting_team) {
        if (player.bat_stats.has_batted) {
            wait_log << batter_scheduler_type << ",1," << player.player_name << "," 
                     << player.expected_stay_duration << "," << player.bat_stats.wait_time << "\n";
        }
    }

    inn1_runs = score.total_runs; inn1_wkts = score.wickets; 
    inn1_overs = score.current_over; inn1_balls = score.current_ball; inn1_extras = score.extras;
    
    target_score.store(score.total_runs + 1);

    pthread_mutex_lock(&print_mutex);
    std::ostringstream oss1;
    oss1 << "\n\n  +" << std::string(BOX_W, '=') << "+\n";
    oss1 << "  |" << pad("  INNINGS BREAK", BOX_W) << "|\n";
    oss1 << "  |" << pad("  " + team1 + " scores: " + std::to_string(inn1_runs) + "/" + std::to_string(inn1_wkts), BOX_W) << "|\n";
    oss1 << "  |" << pad("  " + team2 + " needs " + std::to_string(target_score.load()) + " runs to win.", BOX_W) << "|\n";
    oss1 << "  +" << std::string(BOX_W, '=') << "+\n\n";
    std::string out1 = oss1.str();
    comm_log << out1;
    std::cout << out1;
    pthread_mutex_unlock(&print_mutex);

    // ==========================================
    // INNINGS 2
    // ==========================================
    current_innings.store(2);
    setup_innings(2); 
    print_match_header((*batting_team)[0].player_name, (*batting_team)[1].player_name, (*bowling_team)[6].player_name, (*batting_team)[0].player_team, (*bowling_team)[0].player_team);
    
    for(int i = 0; i < 11; i++) {
        pthread_create(&fielder_threads[i], NULL, fielder_thread, &(*bowling_team)[i]);
    }

    pthread_create(&umpire_t, NULL, umpire_thread, NULL);
    pthread_join(umpire_t, NULL); 

    for(int i = 0; i < 11; i++) {
        pthread_join(fielder_threads[i], NULL);
    }

    // --- LOG WAIT TIMES FOR INNINGS 2 ---
    for (const auto& player : *batting_team) {
        if (player.bat_stats.has_batted) {
            wait_log << batter_scheduler_type << ",2," << player.player_name << "," 
                     << player.expected_stay_duration << "," << player.bat_stats.wait_time << "\n";
        }
    }

    inn2_runs = score.total_runs; inn2_wkts = score.wickets; 
    inn2_overs = score.current_over; inn2_balls = score.current_ball; inn2_extras = score.extras;

    // --- MATCH RESULT USING PRIORITY ENCODING ---
    pthread_mutex_lock(&print_mutex);
    std::ostringstream oss2;
    oss2 << "\n\n  +" << std::string(BOX_W, '=') << "+\n";
    oss2 << "  |" << pad("  MATCH ABANDONED / FINISHED", BOX_W) << "|\n";
    
    if (check_innings_status() == InningsStatus::TARGET_REACHED) {
        oss2 << "  |" << pad("  " + team2 + " WINS by " + std::to_string(10 - score.wickets) + " wickets!", BOX_W) << "|\n";
    } else if (score.total_runs == target_score.load() - 1) {
        oss2 << "  |" << pad("  MATCH TIED! (Super Over required)", BOX_W) << "|\n";
    } else {
        oss2 << "  |" << pad("  " + team1 + " WINS by " + std::to_string(target_score.load() - 1 - score.total_runs) + " runs!", BOX_W) << "|\n";
    }
    oss2 << "  +" << std::string(BOX_W, '=') << "+\n\n";
    std::string out2 = oss2.str();
    comm_log << out2;
    std::cout << out2;
    pthread_mutex_unlock(&print_mutex);
    
    print_final_scorecard();
    print_gantt_chart(); 

    // --- CLEANUP ---
    pthread_mutex_destroy(&pitch.pitch_mutex);
    pthread_mutex_destroy(&score_mutex);
    pthread_mutex_destroy(&print_mutex);
    pthread_mutex_destroy(&gantt_mutex);
    pthread_mutex_destroy(&crease_mutex[0]);
    pthread_mutex_destroy(&crease_mutex[1]);
    sem_destroy(&pitch_empty);
    sem_destroy(&ball_ready_end1);
    sem_destroy(&ball_ready_end2);
    sem_destroy(&crease_capacity);
    sem_destroy(&crease_swap_done);
    
    
    pthread_cond_destroy(&cond_fielder_wake); 
    pthread_mutex_destroy(&mutex_fielder_wake); 
    
    sem_destroy(&fielder_identified);
    sem_destroy(&race_started);
    sem_destroy(&fielder_done);
    sem_destroy(&umpire_sem);
    sem_destroy(&non_striker_done); 

    comm_log.close();
    gantt_file.close();
    wait_log.close();

    std::cout << "Simulation Complete. Check 'commentary_log.txt', 'gantt_log.txt', and 'wait_times.csv'.\n";
    return 0;
}