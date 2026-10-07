#include "random.h"

std::mt19937_64 rng(0);
std::uniform_real_distribution<> dis(0.0, 1.0);

void Generator::build(int cur, int l, int r) {
	if (l == r) {
		tree[cur] = 1 - probabilities[l];
		return;
	}
	int mid = (l + r) >> 1;
	build(cur << 1, l, mid);
	build(cur << 1 | 1, mid + 1, r);
	tree[cur] = tree[cur << 1] * tree[cur << 1 | 1];
}


void Generator::init(const std::vector<double>& probabilities) {
	this->probabilities = probabilities;
	tree.resize(probabilities.size() * 4);
	build(1, 0, probabilities.size() - 1);
}

void Generator::solve(int cur, int l, int r) {
	if (l == r) {
		list.emplace_back(l);
		return;
	}
	const double pl = 1 - tree[cur << 1], pr = 1 - tree[cur << 1 | 1];
	const double sum = 1 - tree[cur << 1];
	double rand = dis(rng);
	int mid = (l + r) >> 1;
	if (rand < pl * pr / sum) {
		solve(cur << 1, l, mid);
		solve(cur << 1 | 1, mid + 1, r);
	} else if (rand >= 1 - pl * (1 - pr) / sum) {
		solve(cur << 1, l, mid);
	} else {
		solve(cur << 1 | 1, mid + 1, r);
	}
}
std::vector<int> Generator::get() {
	list.clear();
	if (dis(rng) >= tree[1]) {
		solve(1, 0, probabilities.size() - 1);
	}
	return list;
}

