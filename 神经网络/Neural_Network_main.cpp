//2151130 Í¯º£²©

#include "Neural_Network.hpp"

int main()
{
	vector<vector<int>>feature = { {1,2,5,7},{3,8,5},{4,9,6,11} };
	vector<double>result = { 1.1,2.2,3.5 };
	Neural_Network<int, double> n1(feature, result);
	n1.PrintStatistics();
}