#pragma once
#include "misc.h"
#include <vector>
#include <unordered_map>

/* A family of hash function is (r, cr, p1, p2)-LSH (p1 > p2, c > 1) 
   if h drawn randomly from the family satisfy the following
   * Pr[h(x) = h(y)] >= p1 when dist(x, y) <= r
   * Pr[h(x) = h(y)] <= p2 when dist(x, y) >= cr
 */

typedef long long hash_t;

struct LSH_parameter {
	double r, c, p1, p2;
};

struct LSH_function {
	point a;
	double w, r;

	LSH_function(double r);
	hash_t operator()(const point &p) const;
};

struct hash_row {
	std::vector<hash_t> row;
	int length;

	hash_row(int length);
	hash_t& operator[](int idx);
	const hash_t& operator[](int idx) const;
};

bool operator<(const hash_row &a, const hash_row &b);
bool operator==(const hash_row &a, const hash_row &b);

template<> struct std::hash<hash_row> {
	std::size_t operator()(const hash_row& row) const;
};

struct LSH_function_row {
	LSH_function_row(int size, double r);

	std::vector<LSH_function> row;
	hash_row operator()(const point &p) const;
};

struct LSH_table_node {
	LSH_table_node *prev, *succ;
	hash_row row;
	point value;

	LSH_table_node(const hash_row &row, const point &p);
};

struct LSH_memory_pool {
	LSH_table_node *nodes;
	int limit, size;
	LSH_table_node *free_ptr;
	LSH_memory_pool();
	~LSH_memory_pool();
	LSH_table_node *new_node(const hash_row &row, const point &p);
	void delete_node(LSH_table_node *ptr);
};

extern LSH_memory_pool LSH_pool;