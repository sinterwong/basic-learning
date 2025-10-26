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

#include <filesystem>
#include <gtest/gtest.h>

using namespace algo_and_ds::graph;
using namespace std;
namespace fs = std::filesystem;

class MSTPerformanceTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  fs::path dataDir = "assets/data/graph";
};

TEST_F(MSTPerformanceTest, Normal) {
  string filename1 = (dataDir / "testG3.txt").string();
  int V1 = 8;

  string filename2 = (dataDir / "testG4.txt").string();
  int V2 = 250;

  string filename3 = (dataDir / "testG5.txt").string();
  int V3 = 1000;

  string filename4 = (dataDir / "testG6.txt").string();
  int V4 = 10000;

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
}
