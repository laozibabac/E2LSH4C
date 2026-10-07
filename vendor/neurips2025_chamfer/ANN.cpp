#include "ANN.h"
#include <cmath>
#include <cassert>
#include <random>

PLEB::PLEB(int row_length, int band_number, double radius)
	: row_length(row_length), band_number(band_number), radius(radius) {
	head_pointers = new std::map<hash_row, LSH_table_node*>[band_number];
	for (int b = 0; b < band_number; ++b) {
		hash_functions.emplace_back(row_length, radius);
	}
}

PLEB::pointer PLEB::insert(const point &p) {
	pointer ptr(band_number);

	for (int b = 0; b < band_number; ++b) {
		hash_row row = hash_functions[b](p);
		LSH_table_node* node = LSH_pool.new_node(row, p);
		
		auto it = head_pointers[b].find(row);
		if (it != head_pointers[b].end()) {
			node->succ = it->second;
			it->second->prev = node;
			it->second = node;
		} else {
			head_pointers[b][row] = node;
		}

		ptr[b] = node;
	}
	return ptr;
}

void PLEB::remove(const pointer &ptr) {
	for (int b = 0; b < band_number; ++b) {
		LSH_table_node *node = ptr[b];
		if (node->prev) {
			node->prev->succ = node->succ;
		} else {
			auto it = head_pointers[b].find(node->row);
			assert(it != head_pointers[b].end());
			assert(it->second == node);

			if (node->succ) {
				it->second = node->succ;
			} else {
				head_pointers[b].erase(it);
			}
		}

		if (node->succ) {
			node->succ->prev = node->prev;
		}

		LSH_pool.delete_node(node);
	}
}

bool PLEB::exist_query(const point &p) const {
	int fail_tries = 0;
	const int fail_limit = 2 * band_number + 5;
	for (int b = 0; b < band_number; ++b) {
		hash_row row = hash_functions[b](p);
		auto it = head_pointers[b].find(row);
		if (it != head_pointers[b].end()) {
			for (LSH_table_node* node = it->second; node; node = node->succ) {
				if (dist(p, node->value) <= radius) return true;
				else fail_tries++;

				if (fail_tries == fail_limit) return false;
			}
		}
	}
	return false;
}

point PLEB::nearest_query(const point &p) const {
	int cnt = 0;
	double min_dist = radius;
	point ret = bad_point_value();
	const int limit = 2 * band_number + 5;
	for (int b = 0; b < band_number; ++b) {
		hash_row row = hash_functions[b](p);
		auto it = head_pointers[b].find(row);
		if (it != head_pointers[b].end()) {
			for (LSH_table_node* node = it->second; node; node = node->succ) {
				double d = dist(p, node->value);
				//printf("!! %.4f %.4f\n", d, radius);
				if (d < min_dist) {
					min_dist = d;
					ret = node->value;
				}
				cnt++;

				if (cnt == limit) return ret;
			}
		}
	}
	return ret;
}

ANN::ANN(int n, double r_min, double r_max, double rho) : approx_ratio(ceil(1 / rho) + 1) {
	size = 0;
	double r = r_min;
	while (r < r_max) {
		radius.push_back(r);
		r *= approx_ratio;
	}
	radius.push_back(r);
	num_r = radius.size();

	//const int row_length = int(log(n) / log(1/0.8)) + 2;
	//const int band_number = int(pow(n, rho)) + 8;

	const int row_length = 5;
	const int band_number = 20;

	for (int i = 0; i < num_r; ++i) {
		PLEB* pleb = new PLEB(row_length, band_number, radius[i]);
		plebs.push_back(pleb);
	}
}

void ANN::copy_hash_function(const ANN* oth) {
	for (int i = 0; i < num_r; ++i) {
		plebs[i]->hash_functions = oth->plebs[i]->hash_functions;
	}
}

ANN::pointer ANN::insert(const point &p) {
	/*printf("ANN.insert %llu ", (unsigned long long) this);
	for (int i = 0; i < DIM; ++i)
		printf("%.4f ", p[i]);
	puts("");*/
	size++;
	ANN::pointer ptrs;
	for (int i = 0; i < num_r; ++i) {
		PLEB::pointer ptr = plebs[i]->insert(p);
		ptrs.push_back(ptr);
	}
	return ptrs;
}

void ANN::remove(const ANN::pointer &ptrs) {
	size--;
	assert(ptrs.size() == num_r);
	for (int i = 0; i < num_r; ++i)
		plebs[i]->remove(ptrs[i]);
}

point ANN::nearest_neighbor(const point &p) const {
	for (int i = 0; i < num_r; ++i) {
		point q = plebs[i]->nearest_query(p);
		if (q.label != -1) {
			//printf("NN %llu %d\n", (unsigned long long) this, i);
			return q;
		}
	}
	//auto hash = plebs[num_r - 1]->hash_functions[0](p);
	//printf("hash: %d %.4f\n", hash[0], plebs[num_r-1]->radius);
	//fprintf(stderr, "??? size = %d\n", size);
	assert(false);
	return bad_point_value();
}

point ANN::query(const point &p) const {
	point neighbor = nearest_neighbor(p);
	return neighbor;
}

count_PLEB::count_PLEB(int n, double radius, double rho) {
	pleb_num = 0;
	for (int i = 1; i <= n; i <<= 1, pleb_num++) ;

	const int row_length = int(log(n) / log(1 / LSH_P2)) + 2;
	const int band_number = 2 * int(pow(n, rho)) + 2;
	for (int j = 0; j < CPLEB_NUM; ++j) {
		for (int i = 0; i < pleb_num; ++i) {
			PLEB* pleb = new PLEB(row_length, band_number, radius);
			plebs[j].push_back(pleb);
		}
	}
}

void count_PLEB::insert(const point &p) {
	
	for (int j = 0; j < CPLEB_NUM; ++j) {
		for (int i = 0; i < pleb_num; ++i) {
			std::uniform_int_distribution<> uni_dist(1, 1<<i);
			if (uni_dist(global_rng) == 1)
				plebs[j][i]->insert(p);
		}
	}
}

double count_PLEB::count(const point &p) {
	double sum = 0;
	for (int j = 0; j < CPLEB_NUM; ++j) {
		for (int i = pleb_num - 1; i >= 0; --i) {
			if (plebs[j][i]->exist_query(p)) {
				sum += 1 << i;
				break;
			}
		}
	}
	return sum / CPLEB_NUM;
}