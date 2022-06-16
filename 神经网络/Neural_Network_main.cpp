//2151130 Í¯º£²©

#include "Neural_Network.hpp"

int main()
{
	vector<vector<int>>feature = { {1,2,3,4,5,6,7,8,9},{5,6,7,8,9,10,11,12,13},{9,10,11,12,13,14,15,16,17} };
	vector<double>result = { 15,18,21,24,27,30,33,36,39 };
	Neural_Network<int,double> n1(feature, result);
	n1.GetDataInfo(1);
	//n1.Normalize();
	n1.GetDataInfo(1);
	//n1.PrintPredict_Result();
	n1.TrainTransform(100000, 0.001);
	n1.PrintPredict_Result();
	n1.PrintTransform();
	vector<int>arr = { 10, 14, 18 };
	cout << n1.Predict(arr) << endl;
}