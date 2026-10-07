#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cmath>
#include <algorithm>
#include <random>
#include <cassert>
#include <functional>
#include <stdexcept>

#include "quad_tree.h"

using namespace std;

static std::mt19937 gen_sample(42);

void seed_quadtree_rng(unsigned long seed) {
    gen_sample.seed(seed);
}

QuadTree::QuadTree(int d, double delta, int max_depth)
    : dim(d), delta(delta), max_depth(max_depth) {
    uniform_real_distribution<double> dist(0.0, delta / 2);
    random_shift.resize(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i) random_shift[static_cast<size_t>(i)] = dist(gen_sample);
    point_buf_.assign(static_cast<size_t>(dim), 0.0);
    origin_buf_.assign(static_cast<size_t>(dim), 0.0);
    repr_buf_.assign(static_cast<size_t>(dim), 0.0);
}

void QuadTree::bind(const CoordAccess& coords) {
    coord_ = coords;
    bound_ = true;
}

void QuadTree::read_id(int id_in_depth, double* dst) const {
    const int d = dim;
    if (id_in_depth >= B_ID_shift) {
        const int b = id_in_depth - B_ID_shift;
        if (coord_.Bf) {
            const float* src = coord_.Bf + static_cast<size_t>(b) * static_cast<size_t>(d);
            for (int i = 0; i < d; ++i) dst[i] = static_cast<double>(src[i]);
        } else {
            const auto& src = (*coord_.Bn)[static_cast<size_t>(b)];
            for (int i = 0; i < d; ++i) dst[i] = src[static_cast<size_t>(i)];
        }
    } else if (coord_.Af) {
        const float* src = coord_.Af + static_cast<size_t>(id_in_depth) * static_cast<size_t>(d);
        for (int i = 0; i < d; ++i) dst[i] = static_cast<double>(src[i]);
    } else {
        const auto& src = (*coord_.An)[static_cast<size_t>(id_in_depth)];
        for (int i = 0; i < d; ++i) dst[i] = src[static_cast<size_t>(i)];
    }
}

void QuadTree::copy_A_row(int index, double* dst) const {
    read_id(index, dst);
}

void QuadTree::grid_bits(const double* point, int origin_id, double width, std::bitset<DIM>& out) {
    read_id(origin_id, origin_buf_.data());
    const double half = width / 2.0;
    out.reset();
    for (int i = 0; i < dim; ++i) {
        const double delta_i = origin_buf_[static_cast<size_t>(i)] - root_origin[static_cast<size_t>(i)];
        const double snapped = std::floor(delta_i / width) * width + root_origin[static_cast<size_t>(i)];
        const int bit = static_cast<int>((point[i] - snapped) / half);
        if (bit != 0 && bit != 1) {
            throw invalid_argument("Error: Invalid index value. Got vec[" + to_string(i) + "] = " +
                                   to_string(bit) + ", must be 0 or 1.");
        }
        out[static_cast<size_t>(i)] = static_cast<bool>(bit);
    }
}

void QuadTree::child_key(int32_t parent, int32_t child, std::bitset<DIM>& out) {
    read_id(nodes_[static_cast<size_t>(child)].point_id, repr_buf_.data());
    const Node& p = nodes_[static_cast<size_t>(parent)];
    grid_bits(repr_buf_.data(), p.point_id, p.width, out);
}

int32_t QuadTree::make_node(int depth, int point_id, double width, int32_t parent) {
    Node n;
    n.parent = parent;
    n.point_id = point_id;
    n.depth = static_cast<int16_t>(depth);
    n.width = width;
    nodes_.push_back(n);
    return static_cast<int32_t>(nodes_.size() - 1);
}

void QuadTree::upgrade(int32_t parent, int32_t existing, const std::bitset<DIM>& existing_key) {
    Node& p = nodes_[static_cast<size_t>(parent)];
    Multi m;
    m.kids.emplace(existing_key, existing);
    const Node& c = nodes_[static_cast<size_t>(existing)];
    if (c.B_count == 0) m.sampler.insert(make_pair(c.depth, c.point_id), static_cast<double>(c.A_count), existing);
    p.multi = 1;
    p.child_or_slot = static_cast<int32_t>(multis_.size());
    multis_.push_back(std::move(m));
}

void QuadTree::link_child(int32_t parent, const std::bitset<DIM>& key, int32_t child) {
    Node& p = nodes_[static_cast<size_t>(parent)];
    if (p.multi) {
        multis_[static_cast<size_t>(p.child_or_slot)].kids.emplace(key, child);
    } else {
        p.child_or_slot = child;
    }
}

int32_t QuadTree::find_child(int32_t parent, const std::bitset<DIM>& key) {
    const Node& p = nodes_[static_cast<size_t>(parent)];
    if (p.multi) {
        const auto& kids = multis_[static_cast<size_t>(p.child_or_slot)].kids;
        auto it = kids.find(key);
        return it == kids.end() ? -1 : it->second;
    }
    if (p.child_or_slot < 0) return -1;
    std::bitset<DIM> have;
    child_key(parent, p.child_or_slot, have);
    return have == key ? p.child_or_slot : -2;
}

bool QuadTree::children_empty(int32_t node) const {
    const Node& n = nodes_[static_cast<size_t>(node)];
    if (n.multi) return multis_[static_cast<size_t>(n.child_or_slot)].kids.empty();
    return n.child_or_slot < 0;
}

void QuadTree::unlink_child(int32_t parent, int32_t child) {
    Node& p = nodes_[static_cast<size_t>(parent)];
    if (p.multi) {
        auto& kids = multis_[static_cast<size_t>(p.child_or_slot)].kids;
        for (auto it = kids.begin(); it != kids.end(); ++it) {
            if (it->second == child) {
                kids.erase(it);
                return;
            }
        }
        throw std::logic_error("prune: child node not found in parent->children map");
    }
    if (p.child_or_slot != child) throw std::logic_error("prune: child node not found in parent->children map");
    p.child_or_slot = -1;
}

void QuadTree::sampler_remove(int32_t parent, int32_t child, double weight) {
    if (!(weight > 1e-4)) return;
    const Node& p = nodes_[static_cast<size_t>(parent)];
    if (!p.multi) return;
    const Node& c = nodes_[static_cast<size_t>(child)];
    multis_[static_cast<size_t>(p.child_or_slot)].sampler.remove(make_pair(c.depth, c.point_id), weight);
}

void QuadTree::sampler_insert(int32_t parent, int32_t child, double weight) {
    if (!(weight > 1e-4)) return;
    const Node& p = nodes_[static_cast<size_t>(parent)];
    if (!p.multi) return;
    const Node& c = nodes_[static_cast<size_t>(child)];
    multis_[static_cast<size_t>(p.child_or_slot)].sampler.insert(make_pair(c.depth, c.point_id), weight, child);
}

void QuadTree::touch_weight(int32_t node, double old_weight, double new_weight) {
    nodes_[static_cast<size_t>(node)].tree_weight = new_weight;
    if (fabs(old_weight - new_weight) > 1e-4) {
        const Node& n = nodes_[static_cast<size_t>(node)];
        auto id = make_pair(static_cast<int>(n.depth), n.point_id);
        weight_tree.remove(id, old_weight);
        weight_tree.insert(id, new_weight, node);
    }
}

template <typename Fn>
void QuadTree::for_each_child(int32_t node, Fn&& fn) const {
    const Node& n = nodes_[static_cast<size_t>(node)];
    if (n.multi) {
        for (const auto& kv : multis_[static_cast<size_t>(n.child_or_slot)].kids) fn(kv.second);
    } else if (n.child_or_slot >= 0) {
        fn(n.child_or_slot);
    }
}

void QuadTree::build_from_bound() {
    if (!bound_) throw std::logic_error("QuadTree::build_from_bound without bind");
    const int nA = coord_.Af ? coord_.nA : (coord_.An ? static_cast<int>(coord_.An->size()) : 0);
    const int nB = coord_.Bf ? coord_.nB : (coord_.Bn ? static_cast<int>(coord_.Bn->size()) : 0);
    vector<double> min_corner(static_cast<size_t>(dim), numeric_limits<double>::max());
    auto scan = [&](int n, bool is_b) {
        for (int r = 0; r < n; ++r) {
            read_id(is_b ? r + B_ID_shift : r, point_buf_.data());
            for (int i = 0; i < dim; ++i)
                min_corner[static_cast<size_t>(i)] = min(min_corner[static_cast<size_t>(i)], point_buf_[static_cast<size_t>(i)]);
        }
    };
    scan(nA, false);
    scan(nB, true);
    root_origin.resize(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i)
        root_origin[static_cast<size_t>(i)] = min_corner[static_cast<size_t>(i)] - random_shift[static_cast<size_t>(i)];

    const size_t guess = static_cast<size_t>(std::max(nA + nB, 1)) * static_cast<size_t>(std::max(max_depth, 1)) + 8;
    nodes_.clear();
    multis_.clear();
    nodes_.reserve(guess);
    root_ = make_node(0, 0, delta, -1);
}

void QuadTree::build(const vector<vector<double>>& A, const vector<vector<double>>& B,
                     const vector<vector<double>>& c_A, const vector<vector<double>>& c_B) {
    CoordAccess c;
    c.An = &A;
    c.Bn = &B;
    c.nA = static_cast<int>(A.size());
    c.nB = static_cast<int>(B.size());
    c.dim = dim;
    bind(c);
    if (c_A.empty() && c_B.empty()) {
        build_from_bound();
    } else {
        root_ = -1;
    }
}

double QuadTree::node_width(int32_t node) const {
    return nodes_[static_cast<size_t>(node)].width;
}

int QuadTree::node_single_A(int32_t node) const {
    return nodes_[static_cast<size_t>(node)].single_A;
}

int32_t QuadTree::sample_node_by_weight(int32_t node) {
    if (node < 0) return -1;
    Node& n = nodes_[static_cast<size_t>(node)];
    if (n.node_total_weight <= 0) return -1;
    std::uniform_real_distribution<> dis(0.0, n.node_total_weight);
    const double random_weight = dis(gen_sample);
    if (random_weight <= 0.0) return -1;
    if (!n.multi) {
        if (n.child_or_slot < 0) return -1;
        const Node& c = nodes_[static_cast<size_t>(n.child_or_slot)];
        const double w = c.B_count == 0 ? static_cast<double>(c.A_count) : 0.0;
        if (!(w > 1e-4) || random_weight > w) return -1;
        return n.child_or_slot;
    }
    std::shared_ptr<BBTNode> sampled = multis_[static_cast<size_t>(n.child_or_slot)].sampler.sample(random_weight);
    return sampled ? sampled->hst_index : -1;
}

double QuadTree::calculateTotalWeight() {
    double total = 0;
    std::function<void(int32_t)> calc = [&](int32_t node) {
        if (node < 0) return;
        total += nodes_[static_cast<size_t>(node)].width * nodes_[static_cast<size_t>(node)].match_count;
        for_each_child(node, [&](int32_t c) { calc(c); });
    };
    calc(root_);
    return total;
}

void QuadTree::Build_Tree_Sampler() {
    weight_tree.clear();
    total_weight = 0;
    std::function<void(int32_t)> build_weights = [&](int32_t node) {
        if (node < 0) return;
        Node& n = nodes_[static_cast<size_t>(node)];
        const double weight = n.width * n.match_count;
        n.tree_weight = weight;
        total_weight += weight;
        weight_tree.insert(make_pair(static_cast<int>(n.depth), n.point_id), weight, node);
        for_each_child(node, [&](int32_t c) { build_weights(c); });
    };
    build_weights(root_);
}

int32_t QuadTree::sample_by_weight() {
    if (root_ < 0 || total_weight <= 0) return -1;
    std::uniform_real_distribution<> dis(0.0, total_weight);
    const double random_weight = dis(gen_sample);
    std::shared_ptr<BBTNode> result = weight_tree.sample(random_weight);
    return result ? result->hst_index : -1;
}

void QuadTree::insert_update_A(int32_t node, int C, int ai) {
    while (node >= 0) {
        Node& n = nodes_[static_cast<size_t>(node)];
        if (n.A_count != 0 && n.is_leaf) n.single_A = ai;
        if (n.A_count == 0) n.single_A = ai;
        else if (n.A_count == 1 && !n.is_leaf) n.single_A = -1;
        n.A_count += 1;
        if (n.parent >= 0 && n.B_count == 0) {
            const double old_weight = static_cast<double>(n.A_count - 1);
            const double new_weight = static_cast<double>(n.A_count);
            sampler_remove(n.parent, node, old_weight);
            sampler_insert(n.parent, node, new_weight);
            nodes_[static_cast<size_t>(n.parent)].node_total_weight += 1;
        }
        if (n.B_count > 0 && C == 1) {
            n.match_count += 1;
            C = 0;
        }
        const double old_weight = n.tree_weight;
        total_weight -= old_weight;
        const double new_weight = n.width * n.match_count;
        total_weight += new_weight;
        touch_weight(node, old_weight, new_weight);
        node = nodes_[static_cast<size_t>(node)].parent;
    }
}

void QuadTree::insert_update_B(int32_t node, int C) {
    nodes_[static_cast<size_t>(node)].B_count += 1;
    if (nodes_[static_cast<size_t>(node)].B_count == 1) {
        nodes_[static_cast<size_t>(node)].match_count = nodes_[static_cast<size_t>(node)].A_count - C;
        C = nodes_[static_cast<size_t>(node)].A_count;
    } else {
        nodes_[static_cast<size_t>(node)].match_count -= C;
        C = 0;
    }
    const int32_t parent = nodes_[static_cast<size_t>(node)].parent;
    if (parent >= 0 && nodes_[static_cast<size_t>(node)].B_count == 1) {
        sampler_remove(parent, node, static_cast<double>(nodes_[static_cast<size_t>(node)].A_count));
        nodes_[static_cast<size_t>(parent)].node_total_weight -= nodes_[static_cast<size_t>(node)].A_count;
    }
    const double old_weight = nodes_[static_cast<size_t>(node)].tree_weight;
    total_weight -= old_weight;
    const double new_weight = nodes_[static_cast<size_t>(node)].width * nodes_[static_cast<size_t>(node)].match_count;
    total_weight += new_weight;
    touch_weight(node, old_weight, new_weight);
    if (parent >= 0) insert_update_B(parent, C);
}

void QuadTree::insert_id(char label, int point_i) {
    if (label == 'A' && (point_i % 200000) == 0) {
        std::cerr << "qt_insert_A i=" << point_i << " nodes=" << nodes_.size()
                  << " multis=" << multis_.size() << std::endl;
    }
    read_id(point_i, point_buf_.data());
    const int32_t leaf = insert_recursive(root_, point_buf_.data(), 0, point_i);
    if (label == 'A') insert_update_A(leaf, 1, point_i);
    else if (label == 'B') insert_update_B(leaf, 0);
    else throw std::invalid_argument("Error: wrong label. Valid labels are 'A' or 'B'.");
}

int32_t QuadTree::insert_recursive(int32_t node, const double* point, int depth, int point_id) {
    if (depth >= max_depth) {
        nodes_[static_cast<size_t>(node)].is_leaf = 1;
        return node;
    }
    std::bitset<DIM> key;
    grid_bits(point, nodes_[static_cast<size_t>(node)].point_id, nodes_[static_cast<size_t>(node)].width, key);
    int32_t child = find_child(node, key);
    if (child == -2) {
        const int32_t existing = nodes_[static_cast<size_t>(node)].child_or_slot;
        std::bitset<DIM> existing_key;
        child_key(node, existing, existing_key);
        upgrade(node, existing, existing_key);
        child = -1;
    }
    if (child < 0) {
        const double new_width = nodes_[static_cast<size_t>(node)].width / 2.0;
        child = make_node(depth + 1, point_id, new_width, node);
        link_child(node, key, child);
    }
    return insert_recursive(child, point, depth + 1, point_id);
}

void QuadTree::remove_id(char label, int point_i) {
    read_id(point_i, point_buf_.data());
    const int32_t leaf = remove_recursive(root_, point_buf_.data(), 0);
    if (label == 'A') delete_update_A(leaf, 1);
    else if (label == 'B') delete_update_B(leaf, 0);
    else throw invalid_argument("Error: wrong label. Valid labels are 'A' or 'B'.");
}

int32_t QuadTree::remove_recursive(int32_t node, const double* point, int depth) {
    if (depth >= max_depth) return node;
    std::bitset<DIM> key;
    grid_bits(point, nodes_[static_cast<size_t>(node)].point_id, nodes_[static_cast<size_t>(node)].width, key);
    const int32_t child = find_child(node, key);
    if (child < 0) throw runtime_error("Point not found in tree");
    return remove_recursive(child, point, depth + 1);
}

void QuadTree::delete_update_A(int32_t node, int C) {
    while (node >= 0) {
        Node& n = nodes_[static_cast<size_t>(node)];
        if (n.A_count == 1) {
            n.single_A = -1;
        } else if (n.A_count == 2) {
            bool found = false;
            for_each_child(node, [&](int32_t c) {
                if (found) return;
                const Node& ch = nodes_[static_cast<size_t>(c)];
                if (ch.A_count > 0) {
                    if (ch.A_count == 1) n.single_A = ch.single_A;
                    found = true;
                }
            });
        } else if (n.A_count == 0) {
            throw std::runtime_error("Node has no A-class points.");
        }
        n.A_count -= 1;
        const int32_t parent = n.parent;
        if (parent >= 0 && n.B_count == 0) {
            const double old_weight = static_cast<double>(n.A_count + 1);
            const double new_weight = static_cast<double>(n.A_count);
            sampler_remove(parent, node, old_weight);
            sampler_insert(parent, node, new_weight);
            nodes_[static_cast<size_t>(parent)].node_total_weight -= 1;
        }
        if (n.B_count > 0 && C == 1) {
            n.match_count -= 1;
            C = 0;
        }
        const double old_w = n.tree_weight;
        total_weight -= old_w;
        const double new_w = n.width * n.match_count;
        total_weight += new_w;
        touch_weight(node, old_w, new_w);
        node = parent;
    }
}

void QuadTree::delete_update_B(int32_t node, int C) {
    nodes_[static_cast<size_t>(node)].B_count--;
    if (nodes_[static_cast<size_t>(node)].B_count == 0) {
        nodes_[static_cast<size_t>(node)].match_count = 0;
        C = nodes_[static_cast<size_t>(node)].A_count;
    } else {
        nodes_[static_cast<size_t>(node)].match_count += C;
        C = 0;
    }
    const int32_t parent = nodes_[static_cast<size_t>(node)].parent;
    if (parent >= 0) {
        if (nodes_[static_cast<size_t>(node)].A_count == 0 && nodes_[static_cast<size_t>(node)].B_count == 0 &&
            children_empty(node)) {
            unlink_child(parent, node);
        }
        delete_update_B(parent, C);
    }
    if (nodes_[static_cast<size_t>(node)].parent >= 0 && nodes_[static_cast<size_t>(node)].B_count == 0 &&
        nodes_[static_cast<size_t>(node)].A_count != 0) {
        sampler_insert(nodes_[static_cast<size_t>(node)].parent, node,
                       static_cast<double>(nodes_[static_cast<size_t>(node)].A_count));
        nodes_[static_cast<size_t>(nodes_[static_cast<size_t>(node)].parent)].node_total_weight +=
            nodes_[static_cast<size_t>(node)].A_count;
    }
    const double old_weight = nodes_[static_cast<size_t>(node)].tree_weight;
    total_weight -= old_weight;
    const double new_weight =
        nodes_[static_cast<size_t>(node)].width * nodes_[static_cast<size_t>(node)].match_count;
    total_weight += new_weight;
    touch_weight(node, old_weight, new_weight);
}

void QuadTree::insert(vector<double>&, char label, int point_i, vector<vector<double>>& A, vector<vector<double>>& B) {
    if (!bound_) {
        CoordAccess c;
        c.An = &A;
        c.Bn = &B;
        c.nA = static_cast<int>(A.size());
        c.nB = static_cast<int>(B.size());
        c.dim = dim;
        bind(c);
    }
    insert_id(label, point_i);
}

void QuadTree::remove(const vector<double>& point, char label, vector<vector<double>>& A, vector<vector<double>>& B) {
    if (!bound_) {
        CoordAccess c;
        c.An = &A;
        c.Bn = &B;
        c.nA = static_cast<int>(A.size());
        c.nB = static_cast<int>(B.size());
        c.dim = dim;
        bind(c);
    }
    if (static_cast<int>(point.size()) != dim) throw std::invalid_argument("remove: dimension mismatch");
    for (int i = 0; i < dim; ++i) point_buf_[static_cast<size_t>(i)] = point[static_cast<size_t>(i)];
    const int32_t leaf = remove_recursive(root_, point_buf_.data(), 0);
    if (label == 'A') delete_update_A(leaf, 1);
    else if (label == 'B') delete_update_B(leaf, 0);
    else throw invalid_argument("Error: wrong label. Valid labels are 'A' or 'B'.");
}
