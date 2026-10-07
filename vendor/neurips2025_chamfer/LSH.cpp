#include "LSH.h"
#include <cmath>
#include <cassert>
#include <random>

LSH_function::LSH_function(double r) : a(rand_unit_vector()), r(r) {
	std::uniform_real_distribution<> uni_dist(0, r);
	w = uni_dist(global_rng);
}

hash_t LSH_function::operator()(const point &p) const {
	hash_t val = (hash_t) floor((dot(p, a) + w) / r);
	if (val >= 0) return val * 2;
	else return -val * 2 + 1;
}

hash_row::hash_row(int l) : length(l) {
	assert(l > 0);
	row.resize(l);
}

hash_t& hash_row::operator[](int idx) {
	assert(0 <= idx && idx < length);
	return row[idx];
}

const hash_t& hash_row::operator[](int idx) const {
	assert(0 <= idx && idx < length);
	return row[idx];
}

bool operator<(const hash_row &a, const hash_row &b) {
	if (a.length != b.length) return a.length < b.length;
	for (int i = 0; i < a.length; ++i) {
		if (a[i] != b[i]) return a[i] < b[i];
	}
	return false;
}

bool operator==(const hash_row &a, const hash_row &b) {
	if (a.length != b.length) return false;
	for (int i = 0; i < a.length; ++i) {
		if (a[i] != b[i]) return false;
	}
	return true;
}

std::size_t std::hash<hash_row>::operator()(const hash_row& row) const {
	const int seed = 4121;
	const int mod = (int) (1e9 + 7);

	size_t val = 0;
	for (int i = 0; i < row.length; ++i) {
		val = (1ll * val * seed + row[i] + 2) % mod;
	}
	return val;
}

LSH_function_row::LSH_function_row(int size, double r) {
	for (int i = 0; i < size; ++i) {
		row.push_back(LSH_function(r));
	}
}

hash_row LSH_function_row::operator()(const point &p) const {
	hash_row val(row.size());
	for (int i = 0; i < row.size(); ++i) {
		val[i] = row[i](p);
	}
	return val;
}

LSH_table_node::LSH_table_node(const hash_row &row, const point &p) : row(row), value(p) {
	prev = nullptr;
	succ = nullptr;
}

LSH_memory_pool::LSH_memory_pool() {
	free_ptr = nullptr;
	limit = 20000000;
	nodes = (LSH_table_node*) malloc(limit * sizeof(LSH_table_node));
	size = 0;
}

LSH_memory_pool::~LSH_memory_pool() {
	free(nodes);
}

LSH_table_node* LSH_memory_pool::new_node(const hash_row &row, const point &p) {
	if (free_ptr) {
		LSH_table_node *ret = free_ptr;
		free_ptr = free_ptr->succ;
		if (free_ptr) free_ptr->prev = nullptr;

		ret->prev = ret->succ = nullptr;
		ret->row = row; ret->value = p;
		return ret;
	} else {
		nodes[size] = LSH_table_node(row, p);
		size++;
		assert(size < limit);
		return &nodes[size-1];
	}
}

void LSH_memory_pool::delete_node(LSH_table_node *ptr) {
	ptr->prev = ptr->succ = nullptr;
	if (!free_ptr) free_ptr = ptr;
	else {
		ptr->succ = free_ptr;
		free_ptr->prev = ptr;
		free_ptr = ptr;
	}
}

LSH_memory_pool LSH_pool;