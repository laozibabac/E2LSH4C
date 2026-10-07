#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <set>

using std::vector;
using std::size_t;

static double euclidean(const vector<double>& p1,
                        const vector<double>& p2) {
    if (p1.size() != p2.size())
        throw std::invalid_argument("Point dimensions are inconsistent.");
    double sum = 0.0;
    for (size_t i = 0; i < p1.size(); ++i) {
        double d = p1[i] - p2[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

class DynamicChamfer {
public:
    DynamicChamfer(const vector<vector<double>>& A_init,
                          const vector<vector<double>>& B_init)
        : A(A_init), B(B_init)
    {
        if (A.empty())
            throw std::invalid_argument("A cannot be empty.");
        dim = A[0].size();
        sumDist = 0.0;
        distSets.reserve(A.size());
        for (size_t i = 0; i < A.size(); ++i) {
            check_dim(A[i]);
            distSets.emplace_back();
            for (const auto& b : B) {
                distSets[i].insert(euclidean(A[i], b));
            }
            if (!distSets[i].empty())
                sumDist += *distSets[i].begin();
        }
    }

    void insert_B(const vector<double>& b_new) {
        check_dim(b_new);
        for (size_t i = 0; i < A.size(); ++i) {
            auto& ms = distSets[i];
            bool wasEmpty = ms.empty();
            double oldMin = wasEmpty ? 0.0 : *ms.begin(); 
            double d = euclidean(A[i], b_new);
            ms.insert(d);
            double newMin = *ms.begin();
            if (wasEmpty) {
                sumDist += newMin; 
            } else {
                sumDist += (newMin - oldMin);
            }
        }
        B.push_back(b_new);
    }

    void delete_B(size_t idx) {
        if (idx >= B.size())
            throw std::out_of_range("delete_B: Index out of bounds.");

        const auto b_old = B[idx];
        for (size_t i = 0; i < A.size(); ++i) {
            auto& ms = distSets[i];
            double oldMin = *ms.begin();
            double d_old = euclidean(A[i], b_old);
            auto it = ms.find(d_old);
            if (it != ms.end())
                ms.erase(it);
            else
                throw std::runtime_error("Distance not found in multiset.");

            double newMin = ms.empty()
                              ? std::numeric_limits<double>::infinity()
                              : *ms.begin();
            sumDist += (newMin - oldMin);
        }
        B.erase(B.begin() + idx);
    }

    void insert_A(const vector<double>& a_new) {
        check_dim(a_new);
        vector<double> dists;
        dists.reserve(B.size());
        for (const auto& b : B) {
            dists.push_back(euclidean(a_new, b));
        }
        std::multiset<double> ms(dists.begin(), dists.end());
        double minD = ms.empty() ? 0.0 : *ms.begin();
        distSets.push_back(std::move(ms));
        A.push_back(a_new);
        sumDist += minD;
    }

    void delete_A(size_t idx) {
        if (idx >= A.size())
            throw std::out_of_range("delete_A: Index out of bounds.");
        sumDist -= (distSets[idx].empty() ? 0.0 : *distSets[idx].begin());
        A.erase(A.begin() + idx);
        distSets.erase(distSets.begin() + idx);
    }

    double current() const {
        return sumDist;
    }

private:
    void check_dim(const vector<double>& p) const {
        if (p.size() != dim)
            throw std::invalid_argument("Point dimensions are inconsistent.");
    }

    vector<vector<double>> A, B;
    size_t dim;
    vector<std::multiset<double>> distSets;
    double sumDist;
};