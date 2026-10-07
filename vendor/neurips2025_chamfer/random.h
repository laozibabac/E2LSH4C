#pragma once
#include<random>
#include<vector>

extern std::mt19937_64 rng;
extern std::uniform_real_distribution<> dis;

class Generator {
  std::vector<double> probabilities;
  std::vector<double> tree;
  void build(int cur, int l, int r);

 public:
  void init(const std::vector<double>& probabilities);
  std::vector<int> list;
  void solve(int cur, int l, int r);
  std::vector<int> get();
};
