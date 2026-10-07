#include "benchmark.h"
#include <cmath>
#include <limits>
#include <stdexcept>

using std::vector;
using std::size_t;

using namespace std;

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

DynamicChamfer::DynamicChamfer(const vector<vector<double>>& A_init,
                               const vector<vector<double>>& B_init)
  : A(A_init), B(B_init)
{
    if (A.empty())
        throw std::invalid_argument("A cannot be empty.");
    dim = A[0].size();
    if (B.empty()){
        minDistAB.assign(A.size(), std::numeric_limits<double>::infinity());
        nnAB   .assign(A.size(), DYNAMIC_CHAMFER_NPOS);
        sumAB = 0.0;
    }else init_direction(A, B, minDistAB, nnAB, sumAB);
    // init_direction(B, A, minDistBA, nnBA, sumBA);
}

void DynamicChamfer::insert_B(const vector<double>& b_new) {
    check_dim(b_new);
    size_t idx = B.size();
    B.push_back(b_new);
    update_insert(A, b_new, idx, minDistAB, nnAB, sumAB);
    // double best = std::numeric_limits<double>::infinity();
    // size_t best_i = DYNAMIC_CHAMFER_NPOS;
    // for (size_t i = 0; i < A.size(); ++i) {
    //     double d = euclidean(b_new, A[i]);
    //     if (d < best) { best = d; best_i = i;}
    // }
    // minDistBA.push_back(best);
    // nnBA.push_back(best_i);
    // sumBA += best;
}

void DynamicChamfer::delete_B(size_t idx) {
    if (idx >= B.size()) throw std::out_of_range("delete_B: Index out of bounds.");
    update_delete(idx, A, B, minDistAB, nnAB, sumAB);
    // sumBA -= minDistBA[idx];
    // minDistBA.erase(minDistBA.begin() + idx);
    // nnBA    .erase(nnBA    .begin() + idx);
    B.erase(B.begin() + idx);
}

void DynamicChamfer::insert_A(const vector<double>& a_new) {
    check_dim(a_new);
    size_t idx = A.size();
    A.push_back(a_new);
    // update_insert(B, a_new, idx, minDistBA, nnBA, sumBA);
    double best = std::numeric_limits<double>::infinity();
    size_t best_j = DYNAMIC_CHAMFER_NPOS;
    for (size_t j = 0; j < B.size(); ++j) {
        double d = euclidean(a_new, B[j]);
        if (d < best) { best = d; best_j = j; }
    }
    minDistAB.push_back(best);
    nnAB.push_back(best_j);
    sumAB += best;
}

void DynamicChamfer::delete_A(size_t idx) {
    if (idx >= A.size()) throw std::out_of_range("delete_A: Index out of bounds.");
    // update_delete(idx, B, A, minDistBA, nnBA, sumBA);
    sumAB -= minDistAB[idx];
    minDistAB.erase(minDistAB.begin() + idx);
    nnAB    .erase(nnAB    .begin() + idx);
    A.erase(A.begin() + idx);
}

double DynamicChamfer::current() const {
    return sumAB; // + sumBA;
}

void DynamicChamfer::check_dim(const vector<double>& p) const {
    if (p.size() != dim)
        throw std::invalid_argument("Insertion point dimensions are inconsistent.");
}

void DynamicChamfer::init_direction(const vector<vector<double>>& S,
                                    const vector<vector<double>>& T,
                                    vector<double>& minDist,
                                    vector<size_t>& nnIdx,
                                    double& sumDist)
{
    size_t n = S.size();
    minDist.assign(n, std::numeric_limits<double>::infinity());
    nnIdx .assign(n, DYNAMIC_CHAMFER_NPOS);
    sumDist = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < T.size(); ++j) {
            double d = euclidean(S[i], T[j]);
            if (d < minDist[i]) {
                minDist[i] = d;
                nnIdx[i]   = j;
            }
        }
        sumDist += minDist[i];
    }
}

void DynamicChamfer::update_insert(const vector<vector<double>>& S,
                                   const vector<double>& t_new,
                                   size_t idx,
                                   vector<double>& minDist,
                                   vector<size_t>& nnIdx,
                                   double& sumDist)
{
    for (size_t i = 0; i < S.size(); ++i) {
        double d = euclidean(S[i], t_new);
        if (d < minDist[i]) {
            if (minDist[i] == std::numeric_limits<double>::infinity()) {
                sumDist += d;  
            } else {
                sumDist += (d - minDist[i]); 
            }
            minDist[i] = d;
            nnIdx[i]   = idx;
        }
    }
}

void DynamicChamfer::update_delete(size_t idx_del,
                                   const vector<vector<double>>& S,
                                   const vector<vector<double>>& T_old,
                                   vector<double>& minDist,
                                   vector<size_t>& nnIdx,
                                   double& sumDist)
{
    vector<size_t> to_recompute;
    for (size_t i = 0; i < S.size(); ++i) {
        if (nnIdx[i] == idx_del) {
            sumDist    -= minDist[i];
            minDist[i]  = std::numeric_limits<double>::infinity();
            nnIdx[i]    = DYNAMIC_CHAMFER_NPOS;
            to_recompute.push_back(i);
        }
    }
    for (size_t i = 0; i < S.size(); ++i) {
        if (nnIdx[i] != DYNAMIC_CHAMFER_NPOS && nnIdx[i] > idx_del)
            --nnIdx[i];
    }
    for (size_t i : to_recompute) {
        for (size_t j = 0; j < T_old.size(); ++j) {
            if (j == idx_del) continue;
            double d = euclidean(S[i], T_old[j]);
            size_t new_j = (j < idx_del ? j : j - 1);
            if (d < minDist[i]) {
                minDist[i] = d;
                nnIdx[i]   = new_j;
            }
        }
        sumDist += minDist[i];
    }
}

