#include "balanced_binary_tree.h"
#include "quad_tree.h"

using namespace std;

// TODO: When building, sort by weight first, then by id. The smaller id in the left.
// Get the height of a node
int AVLTree::getHeight(const shared_ptr<BBTNode>& node) {
    return node ? node->height : 0;
}

// Get the balance factor of a node
int AVLTree::getBalanceFactor(const shared_ptr<BBTNode>& node) {
    return node ? getHeight(node->left) - getHeight(node->right) : 0;
}

// Insert a node with the given ID and weight
void AVLTree::insert(pair<int, int> id, double weight, int32_t hst_index) {
    if(weight > 1e-4) root = insertNode(root, id, weight, hst_index, nullptr); // Pass nullptr as root's parent
}

// Remove a node with the given ID
void AVLTree::remove(pair<int, int> id, double weight) {
    root = deleteNode(root, id, weight);
}

// Update a node in the AVL Tree
// void AVLTree::update(pair<int, vector<int>> old_id, double old_weight, double new_weight) {
//     updateNode(root, old_id, old_weight, new_weight);
// }

// Print the AVL Tree in order
void AVLTree::clear() {
    root = nullptr;
}

void AVLTree::printInOrder(shared_ptr<BBTNode> node, double& total_weight, int& count) const {
    if (node) {
        printInOrder(node->left, total_weight, count);
        
        total_weight += node->weight;
        count++;

        cout << "Node (" << node->node_id.first << " " << node->node_id.second << "), Weight: " << node->weight << endl;

        printInOrder(node->right, total_weight, count);
    }
}

std::shared_ptr<BBTNode> AVLTree::sample(double random_weight) const {
    if (!root || random_weight <= 0) return nullptr;

    std::shared_ptr<BBTNode> current = root;

    while (current) {
        double left_sum = current->left ? current->left->subtree_weight_sum : 0.0;

        if (random_weight < left_sum) {
            current = current->left;
        } else if (random_weight <= left_sum + current->weight) {
            return current;
        } else {
            random_weight -= (left_sum + current->weight);
            current = current->right;
        }
        // std::cerr << "Visiting node: weight=" << current->weight 
        // << ", left_sum=" << left_sum 
        // << ", subtree_weight_sum=" << current->subtree_weight_sum 
        // << ", random_weight=" << random_weight << std::endl;

    }

    return nullptr;
}    

