#include "datatype.h"

point::point(const std::vector<double>& v, int label) : v(v), label(label) { }

double& point::operator[](int idx) {
    return v[idx];
}

const double& point::operator[](int idx) const {
    return v[idx];
}

int point::size() const {
    return v.size();
}

double point::distance(const point& other) const {
    double sum = 0.0;
    for (int i = 0; i < size(); ++i) {
        double d = v[i] - other.v[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

bool operator==(const point &a, const point &b) {
    if (a.label != b.label || a.v.size() != b.v.size()) return false;
    for (int i = 0; i < a.v.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

std::size_t std::hash<point>::operator()(const point& p) const {
    const int seed = 4121;
    const int mod = 1e9 + 7;
    size_t val = 0;
    for (double d : p.v) {
        val = (1ll * val * seed + static_cast<int>(d + 2)) % mod;
    }
    return val;
}

std::ostream& operator<<(std::ostream& os, const point& p) {
    os << "point(label=" << p.label << ", v=[";
    for (size_t i = 0; i < p.v.size(); ++i) {
        os << p.v[i];
        if (i != p.v.size() - 1) os << ", ";
    }
    os << "])";
    return os;
}

std::vector<point> convert_to_points(const std::vector<std::vector<double>>& B) {
    std::vector<point> result;
    for (int i = 0; i < B.size(); ++i) {
        point p(B[i], i);
        result.push_back(p);
    }
    return result;
}
