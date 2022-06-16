//2151130 Í¯º£²©

#include "Neural_Network.hpp"

int main()
{
	vector<vector<int>>feature = { {1,2,3,4},{5,6,7,8},{9,10,11,12} };
	vector<double>result = { 15,18,21,24 };
	Neural_Network<int,double> n1(feature, result);
	n1.GetDataInfo(1);
	//n1.Normalize();
	//n1.GetDataInfo(1);
	n1.PrintPredict_Result();
	n1.TrainTransform(5000, 0.07);
	n1.PrintPredict_Result();
	n1.PrintTransform();
	vector<int>arr = { 5, 9, 13 };
	cout << n1.Predict(arr) << endl;
}