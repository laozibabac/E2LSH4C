#ifndef BALANCED_BINARY_TREE_H
#define BALANCED_BINARY_TREE_H

#include <iostream>
#include <memory>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <vector>

// Node structure for Balanced Binary Tree (AVL Tree)
class BBTNode {
public:
    std::pair<int, int>       node_id; // Quadtree node's unique ID
    double                       weight; // Quadtree node's weight
    double                       subtree_weight_sum; // Sum of weights in this subtree
    int32_t                      hst_index; // index into QuadTree::nodes_
    std::shared_ptr<BBTNode>          left, right, parent; // Left, Right children and Parent pointer
    int height; // Height of the node (used for balancing)
    
    BBTNode(std::pair<int, int> id, double w, int32_t hst) : 
        node_id(id), weight(w), subtree_weight_sum(w), hst_index(hst),
        height(1), left(nullptr), right(nullptr), parent(nullptr) {}
};

// AVL Tree structure
class AVLTree {
public:
    std::shared_ptr<BBTNode> root;
    AVLTree() : root(nullptr) {}

    void insert(std::pair<int, int> id, double weight, int32_t hst_index);
    void remove(std::pair<int, int> id, double weight);
    void clear();
    void printInOrder(std::shared_ptr<BBTNode> node, double& total_weight, int& count) const;
    std::shared_ptr<BBTNode> getRoot() const { return root; }

    std::shared_ptr<BBTNode> sample(double random_weight) const;

    bool contains(std::pair<int, int> id, double weight) {
        auto node = root;
        while (node) {
            if (isLess(weight, id, node->weight, node->node_id)) {
                node = node->left;
            }
            else if (isGreater(weight, id, node->weight, node->node_id)) {
                node = node->right;
            }
            else {
                return true;
            }
        }
        return false;
    }
    

private:
    // Helper functions for AVL Tree
    int getHeight(const std::shared_ptr<BBTNode>& node);
    int getBalanceFactor(const std::shared_ptr<BBTNode>& node);
    std::shared_ptr<BBTNode> rightRotate(std::shared_ptr<BBTNode> y) {
        std::shared_ptr<BBTNode> x = y->left;
        std::shared_ptr<BBTNode> T2 = x->right;

        // Perform rotation
        x->right = y;
        y->left = T2;

        // Update parents
        x->parent = y->parent;
        y->parent = x;
        if (T2) T2->parent = y;

        // Update heights
        y->height = std::max(getHeight(y->left), getHeight(y->right)) + 1;
        x->height = std::max(getHeight(x->left), getHeight(x->right)) + 1;

        // Update subtree weights
        y->subtree_weight_sum = y->weight;
        if (y->left) y->subtree_weight_sum += y->left->subtree_weight_sum;
        if (y->right) y->subtree_weight_sum += y->right->subtree_weight_sum;
        
        x->subtree_weight_sum = x->weight;
        if (x->left) x->subtree_weight_sum += x->left->subtree_weight_sum;
        if (x->right) x->subtree_weight_sum += x->right->subtree_weight_sum;

        return x;
    }

    std::shared_ptr<BBTNode> leftRotate(std::shared_ptr<BBTNode> x) {
        std::shared_ptr<BBTNode> y = x->right;
        std::shared_ptr<BBTNode> T2 = y->left;

        // Perform rotation
        y->left = x;
        x->right = T2;

        // Update parents
        y->parent = x->parent;
        x->parent = y;
        if (T2) T2->parent = x;

        // Update heights
        x->height = std::max(getHeight(x->left), getHeight(x->right)) + 1;
        y->height = std::max(getHeight(y->left), getHeight(y->right)) + 1;

        // Update subtree weights
        x->subtree_weight_sum = x->weight;
        if (x->left) x->subtree_weight_sum += x->left->subtree_weight_sum;
        if (x->right) x->subtree_weight_sum += x->right->subtree_weight_sum;
        
        y->subtree_weight_sum = y->weight;
        if (y->left) y->subtree_weight_sum += y->left->subtree_weight_sum;
        if (y->right) y->subtree_weight_sum += y->right->subtree_weight_sum;

        return y;
    }

    bool isLess (double w1, const std::pair<int, int>& id1,
        double w2, const std::pair<int, int>& id2) {
        if (w1 < w2) return true;
        if (w1 > w2) return false;
        return id1 < id2;
    };
    bool isGreater (double w1, const std::pair<int, int>& id1,
            double w2, const std::pair<int, int>& id2) {
        if (w1 > w2) return true;
        if (w1 < w2) return false;
        return id1 > id2;
    };

    std::shared_ptr<BBTNode> insertNode(
        std::shared_ptr<BBTNode>& node,
        const std::pair<int, int>& id,
        double weight,
        int32_t hst_index,
        std::shared_ptr<BBTNode> parent)
    {
        if (weight <= 1e-4) return node;
        struct Depth {
            static int& n() { static int d = 0; return d; }
            Depth() {
                int d = ++n();
                if (d >= 64 && (d & (d - 1)) == 0) {
                    std::fprintf(stderr, "avl_insert_depth %d\n", d);
                    std::fflush(stderr);
                }
            }
            ~Depth() { --n(); }
        } depth_guard;
    
        if (!node) {
            auto newNode = std::make_shared<BBTNode>(id, weight, hst_index);
            newNode->parent = parent;
            return newNode;
        }
    
        if (isLess(weight, id, node->weight, node->node_id)) {
            node->left = insertNode(node->left, id, weight, hst_index, node);
        }
        else if (isGreater(weight, id, node->weight, node->node_id)) {
            node->right = insertNode(node->right, id, weight, hst_index, node);
        }
        else {
            return node;
        }
    
        node->height = 1 + std::max(getHeight(node->left), getHeight(node->right));
        node->subtree_weight_sum = node->weight;
        if (node->left)  node->subtree_weight_sum += node->left->subtree_weight_sum;
        if (node->right) node->subtree_weight_sum += node->right->subtree_weight_sum;
    
        int balance = getBalanceFactor(node);
    
        // LL
        if (balance > 1 &&
            isLess(weight, id, node->left->weight, node->left->node_id))
        {
            return rightRotate(node);
        }
        // RR
        if (balance < -1 &&
            isGreater(weight, id, node->right->weight, node->right->node_id))
        {
            return leftRotate(node);
        }
        // LR
        if (balance > 1 &&
            isGreater(weight, id, node->left->weight, node->left->node_id))
        {
            node->left = leftRotate(node->left);
            return rightRotate(node);
        }
        // RL
        if (balance < -1 &&
            isLess(weight, id, node->right->weight, node->right->node_id))
        {
            node->right = rightRotate(node->right);
            return leftRotate(node);
        }
    
        return node;
    }
    

    std::shared_ptr<BBTNode> deleteNode(
        std::shared_ptr<BBTNode>& node,
        std::pair<int, int> id,
        double weight)
    {
        if (!node || weight <= 1e-4) return node;
        struct Depth {
            static int& n() { static int d = 0; return d; }
            Depth() {
                int d = ++n();
                if (d >= 64 && (d & (d - 1)) == 0) {
                    std::fprintf(stderr, "avl_delete_depth %d\n", d);
                    std::fflush(stderr);
                }
            }
            ~Depth() { --n(); }
        } depth_guard;
    
        if (isLess(weight, id, node->weight, node->node_id)) {
            node->left = deleteNode(node->left, id, weight);
        }
        else if (isGreater(weight, id, node->weight, node->node_id)) {
            node->right = deleteNode(node->right, id, weight);
        }
        else {
            if (!node->left || !node->right) {
                std::shared_ptr<BBTNode> temp = node->left ? node->left : node->right;
                if (!temp) {
                    temp = node;
                    node = nullptr;
                } else {
                    *node = *temp;
                }
            } else {
                std::shared_ptr<BBTNode> temp = minValueNode(node->right);
                node->node_id = temp->node_id;
                node->weight  = temp->weight;
                node->hst_index = temp->hst_index;
                node->right = deleteNode(node->right, temp->node_id, temp->weight);
            }
        }
    
        if (!node) return node;
    
        node->height = 1 + std::max(getHeight(node->left), getHeight(node->right));
        node->subtree_weight_sum = node->weight;
        if (node->left)  node->subtree_weight_sum += node->left->subtree_weight_sum;
        if (node->right) node->subtree_weight_sum += node->right->subtree_weight_sum;
    
        int balance = getBalanceFactor(node);
        // LL 
        if (balance > 1 && getBalanceFactor(node->left) >= 0) {
            return rightRotate(node);
        }
        // LR 
        if (balance > 1 && getBalanceFactor(node->left) < 0) {
            node->left = leftRotate(node->left);
            return rightRotate(node);
        }
        // RR 
        if (balance < -1 && getBalanceFactor(node->right) <= 0) {
            return leftRotate(node);
        }
        // RL 
        if (balance < -1 && getBalanceFactor(node->right) > 0) {
            node->right = rightRotate(node->right);
            return leftRotate(node);
        }
    
        return node;
    }    

    std::shared_ptr<BBTNode> minValueNode(std::shared_ptr<BBTNode> node) {
        std::shared_ptr<BBTNode> current = node;
        while (current->left) {
            current = current->left;
        }
        return current;
    }
};

#endif
