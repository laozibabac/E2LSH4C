#pragma once
#include "LSH.h"
#include <map>

#define LSH_P2 0.8

struct PLEB {
	typedef std::vector<LSH_table_node*> pointer;

	std::vector<LSH_function_row> hash_functions;

	std::map<hash_row, LSH_table_node*> *head_pointers;

	int row_length, band_number;
	double radius;

	//std::vector<LSH_table_node> nodes;

	PLEB(int row_length, int band_number, double radius);

	pointer insert(const point &p);
	void remove(const pointer &ptr);
	bool exist_query(const point &p) const;
	point nearest_query(const point &p) const;
};


struct ANN {
	typedef std::vector<PLEB::pointer> pointer;
	
	std::vector<PLEB*> plebs;
	std::vector<double> radius;
	int num_r, size;
	const int approx_ratio;

	ANN(int n, double r_min, double r_max, double rho);

	void copy_hash_function(const ANN* oth);
	pointer insert(const point &p);
	void remove(const pointer &ptrs);
	point nearest_neighbor(const point &p) const;
	point query(const point &p) const;
};

#define CPLEB_NUM 3

struct count_PLEB {
	std::vector<PLEB*> plebs[CPLEB_NUM];
	int pleb_num;

	count_PLEB(int n, double radius, double rho);
	void insert(const point &p);
	double count(const point &p);
};