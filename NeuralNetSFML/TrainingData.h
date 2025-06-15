#pragma once
#include<vector>
#include<string>
#include<fstream>
using namespace std;

struct TrainingData {
	ifstream file;

	TrainingData(const string& filename);
	bool isEof() const;
	void getTopology(vector<unsigned>& topology);
	bool getNextInputs(vector<float>& inputVals);
	bool getTargetOutputs(vector<float>& targetVals);
	void reset();
};