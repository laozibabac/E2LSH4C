#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <bitset>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <cmath>
#include "balanced_binary_tree.h"
#include "dataset_config.h"

// Seed QT sampling + random_shift RNG (native compare driver).
void seed_quadtree_rng(unsigned long seed);

// Either a flat float matrix (f32bin) or nested doubles. float->double is exact
// for values that were loaded from f32, so both feeds produce the same tree.
struct CoordAccess {
    const float* Af = nullptr;
    const float* Bf = nullptr;
    const std::vector<std::vector<double>>* An = nullptr;
    const std::vector<std::vector<double>>* Bn = nullptr;
    int nA = 0;
    int nB = 0;
    int dim = 0;
};

class QuadTree {
public:
    QuadTree(int d, double delta, int max_depth);
    void bind(const CoordAccess& coords);
    void build_from_bound();
    void build(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B,
               const std::vector<std::vector<double>>& c_A, const std::vector<std::vector<double>>& c_B);
    // Returns a node index, or -1.
    int32_t sample_by_weight();
    int32_t sample_node_by_weight(int32_t node);
    double node_width(int32_t node) const;
    int node_single_A(int32_t node) const;
    void copy_A_row(int index, double* dst) const;
    double calculateTotalWeight();
    void Build_Tree_Sampler();
    void insert_id(char label, int point_i);
    void remove_id(char label, int point_i);
    void insert(std::vector<double>& point, char label, int point_i,
                std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B);
    void remove(const std::vector<double>& point, char label,
                std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B);

    double total_weight = 0.0;
    AVLTree weight_tree;

private:
    struct Node {
        int32_t parent = -1;
        // Unary child index, or slot in multis_ when multi != 0.
        int32_t child_or_slot = -1;
        int32_t point_id = 0;
        int32_t single_A = -1;
        int32_t A_count = 0;
        int32_t B_count = 0;
        int32_t match_count = 0;
        int16_t depth = 0;
        uint8_t is_leaf = 0;
        uint8_t multi = 0;
        double width = 0.0;
        double tree_weight = 0.0;
        double node_total_weight = 0.0;
    };
    struct Multi {
        std::unordered_map<std::bitset<DIM>, int32_t> kids;
        AVLTree sampler;
    };

    int dim = 0;
    double delta = 0.0;
    int max_depth = 0;
    int32_t root_ = -1;
    CoordAccess coord_{};
    bool bound_ = false;
    std::vector<double> random_shift;
    std::vector<double> root_origin;
    std::vector<double> point_buf_;
    std::vector<double> origin_buf_;
    std::vector<double> repr_buf_;
    std::vector<Node> nodes_;
    std::vector<Multi> multis_;

    void read_id(int id_in_depth, double* dst) const;
    void grid_bits(const double* point, int origin_id, double width, std::bitset<DIM>& out);
    void child_key(int32_t parent, int32_t child, std::bitset<DIM>& out);
    int32_t make_node(int depth, int point_id, double width, int32_t parent);
    void upgrade(int32_t parent, int32_t existing, const std::bitset<DIM>& existing_key);
    void link_child(int32_t parent, const std::bitset<DIM>& key, int32_t child);
    int32_t find_child(int32_t parent, const std::bitset<DIM>& key);
    bool children_empty(int32_t node) const;
    void unlink_child(int32_t parent, int32_t child);
    void sampler_remove(int32_t parent, int32_t child, double weight);
    void sampler_insert(int32_t parent, int32_t child, double weight);
    void touch_weight(int32_t node, double old_weight, double new_weight);
    int32_t insert_recursive(int32_t node, const double* point, int depth, int point_id);
    int32_t remove_recursive(int32_t node, const double* point, int depth);
    void insert_update_A(int32_t node, int C, int ai);
    void insert_update_B(int32_t node, int C);
    void delete_update_A(int32_t node, int C);
    void delete_update_B(int32_t node, int C);
    template <typename Fn>
    void for_each_child(int32_t node, Fn&& fn) const;
};
