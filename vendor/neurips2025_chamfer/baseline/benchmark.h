#ifndef DYNAMIC_CHAMFER_H
#define DYNAMIC_CHAMFER_H

#include <vector>
#include <cstddef>

static constexpr size_t DYNAMIC_CHAMFER_NPOS = static_cast<size_t>(-1);

static double euclidean(const std::vector<double>& p1,
                 const std::vector<double>& p2);

class DynamicChamfer {
public:
    DynamicChamfer(const std::vector<std::vector<double>>& A_init,
                   const std::vector<std::vector<double>>& B_init);

    void insert_B(const std::vector<double>& b_new);
    void delete_B(size_t idx);
    void insert_A(const std::vector<double>& a_new);
    void delete_A(size_t idx);
    double current() const;

private:
    size_t dim;
    std::vector<std::vector<double>> A, B;
    std::vector<double> minDistAB, minDistBA;
    std::vector<size_t> nnAB, nnBA;
    double sumAB = 0.0, sumBA = 0.0;

    void check_dim(const std::vector<double>& p) const;
    void init_direction(const std::vector<std::vector<double>>& S,
                        const std::vector<std::vector<double>>& T,
                        std::vector<double>& minDist,
                        std::vector<size_t>& nnIdx,
                        double& sumDist);
    void update_insert(const std::vector<std::vector<double>>& S,
                       const std::vector<double>& t_new,
                       size_t idx,
                       std::vector<double>& minDist,
                       std::vector<size_t>& nnIdx,
                       double& sumDist);
    void update_delete(size_t idx_del,
                       const std::vector<std::vector<double>>& S,
                       const std::vector<std::vector<double>>& T_old,
                       std::vector<double>& minDist,
                       std::vector<size_t>& nnIdx,
                       double& sumDist);
};

#endif // DYNAMIC_CHAMFER_H