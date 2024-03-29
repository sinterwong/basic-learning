/**
 * @file test_mst_performance.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-02-05
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "kruskal_mst.hpp"
#include "lazy_prim_mst.hpp"
#include "prim_mst.hpp"
#include "weighted_read_graph.hpp"
#include "weighted_sparse_graph.hpp"
#include <string>

using namespace algo_and_ds::graph;
using namespace std;

int main() {

  string filename1 = "/home/wangxt/workspace/projects/basic-learning/src/"
                     "algo_and_ds/graph/testG3.txt";
  int V1 = 8;

  string filename2 = "/home/wangxt/workspace/projects/basic-learning/src/"
                     "algo_and_ds/graph/testG4.txt";
  int V2 = 250;

  string filename3 = "/home/wangxt/workspace/projects/basic-learning/src/"
                     "algo_and_ds/graph/testG5.txt";
  int V3 = 1000;

  string filename4 = "/home/wangxt/workspace/projects/basic-learning/src/"
                     "algo_and_ds/graph/testG6.txt";
  int V4 = 10000;

  // 文件读取
  SparseGraph<double> g1 = SparseGraph<double>(V1, false);
  ReadGraph<SparseGraph<double>, double> readGraph1(g1, filename1);
  cout << filename1 << " load successfully." << endl;

  SparseGraph<double> g2 = SparseGraph<double>(V2, false);
  ReadGraph<SparseGraph<double>, double> readGraph2(g2, filename2);
  cout << filename2 << " load successfully." << endl;

  SparseGraph<double> g3 = SparseGraph<double>(V3, false);
  ReadGraph<SparseGraph<double>, double> readGraph3(g3, filename3);
  cout << filename3 << " load successfully." << endl;

  SparseGraph<double> g4 = SparseGraph<double>(V4, false);
  ReadGraph<SparseGraph<double>, double> readGraph4(g4, filename4);
  cout << filename4 << " load successfully." << endl;

  cout << endl;

  clock_t startTime, endTime;

  // Test Lazy Prim MST
  cout << "Test Lazy Prim MST:" << endl;

  startTime = clock();
  LazyPrimMST<SparseGraph<double>, double> lazyPrimMST1(g1);
  endTime = clock();
  cout << "Test for G1: " << (double)(endTime - startTime) / CLOCKS_PER_SEC
       << " s." << endl;

  startTime = clock();
  LazyPrimMST<SparseGraph<double>, double> lazyPrimMST2(g2);
  endTime = clock();
  cout << "Test for G2: " << (double)(endTime - startTime) / CLOCKS_PER_SEC
       << " s." << endl;

  startTime = clock();
  LazyPrimMST<SparseGraph<double>, double> lazyPrimMST3(g3);
  endTime = clock();
  cout << "Test for G3: " << (double)(endTime - startTime) / CLOCKS_PER_SEC
       << " s." << endl;

  startTime = clock();
  LazyPrimMST<SparseGraph<double>, double> lazyPrimMST4(g4);
  endTime = clock();
  cout << "Test for G4: " << (double)(endTime - startTime) / CLOCKS_PER_SEC
       << " s." << endl;

  cout << endl;

  // Test Prim MST
  cout << "Test Prim MST:" << endl;

  startTime = clock();
  PrimMST<SparseGraph<double>, double> PrimMST1(g1);
  endTime = clock();
  cout << "Test for G1: " << (double)(endTime - startTime) / CLOCKS_PER_SEC
       << " s." << endl;

  startTime = clock();
  PrimMST<SparseGraph<double>, double> PrimMST2(g2);
  endTime = clock();
  cout << "Test for G2: " << (double)(endTime - startTime) / CLOCKS_PER_SEC
       << " s." << endl;

  startTime = clock();
  PrimMST<SparseGraph<double>, double> PrimMST3(g3);
  endTime = clock();
  cout << "Test for G3: " << (double)(endTime - startTime) / CLOCKS_PER_SEC
       << " s." << endl;

  startTime = clock();
  PrimMST<SparseGraph<double>, double> PrimMST4(g4);
  endTime = clock();
  cout << "Test for G4: " << (double)(endTime - startTime) / CLOCKS_PER_SEC
       << " s." << endl;

  cout << endl;

  // Test Prim MST
  cout << "Test Kruskal MST:" << endl;

  startTime = clock();
  KruskalMST<SparseGraph<double>, double> KruskalMST1(g1);
  endTime = clock();
  cout << "Test for G1: " << (double)(endTime - startTime) / CLOCKS_PER_SEC
       << " s." << endl;

  startTime = clock();
  KruskalMST<SparseGraph<double>, double> KruskalMST2(g2);
  endTime = clock();
  cout << "Test for G2: " << (double)(endTime - startTime) / CLOCKS_PER_SEC
       << " s." << endl;

  startTime = clock();
  KruskalMST<SparseGraph<double>, double> KruskalMST3(g3);
  endTime = clock();
  cout << "Test for G3: " << (double)(endTime - startTime) / CLOCKS_PER_SEC
       << " s." << endl;

  startTime = clock();
  KruskalMST<SparseGraph<double>, double> KruskalMST4(g4);
  endTime = clock();
  cout << "Test for G4: " << (double)(endTime - startTime) / CLOCKS_PER_SEC
       << " s." << endl;

  cout << endl;

  return 0;
}