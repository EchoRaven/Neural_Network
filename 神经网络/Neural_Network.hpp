//2151130 童海博

#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#include <iomanip>
#include <algorithm>
#define UI unsigned int
using namespace std;

//通用函数
//获取数组平均值(传什么类型的数值就返回什么类型的数值)
template <typename type>
double GetMeanOfVector(const vector<type>&arr)
{
	vector<type>temp(arr);
	double total = 0;
	typename vector<type>::iterator it = temp.begin();
	for (; it != temp.end(); it++)
		total += double(*it);
	return total / temp.size();
}

//获取数组的中位数
template<typename type>
double GetMiddleOfVector(const vector<type>&arr)
{
	vector<type>temp(arr);
	sort(temp.begin(), temp.end());
	if (temp.size() % 2 == 0)
		return double((temp[temp.size() / 2] + temp[temp.size() / 2 - 1])) / 2;
	return double(temp[temp.size() / 2]);
}

//获取数组的方差
template<typename type>
double GetSigmaOfVector(const vector<type>& arr)
{
	double mean = GetMeanOfVector(arr);
	double sigma = 0;
	double total = 0;
	for (typename vector<type>::const_iterator it = arr.begin(); it != arr.end(); it++)
	{
		total += (double(*it) - mean) * (double(*it) - mean);
	}
	sigma = sqrt(total / arr.size());
	return sigma;
}

//打印数组
template<typename type>
void PrintVector(vector<type>arr)
{
	for (typename vector<type>::iterator it = arr.begin(); it != arr.end(); it++)
	{
		cout << setiosflags(ios::left) << setw(8) << fixed << setprecision(2) << *it;
	}
	cout << endl;
}


//模板类实现
template<typename Intype, typename Outtype>
class Neural_Network
{
private:
	//神经网络输入端参数
	vector<vector<double>> feature;

	//附加参数信息
	//列表均值
	vector<double> mean_f;

	//列表中位数
	vector<double> middle_f;

	//列表的方差
	vector<double>sigma_f;

	//神经网络输出端参数
	vector<double> result;

	//附加参数信息
	//列表均值
	double mean_r;

	//列表中位数
	double middle_r;

	//列表方差
	double sigma_r;

	//特征集维度
	UI feature_num;

	//数据数量
	UI data_num;

public:
	//默认构造函数
	Neural_Network() {};

	//有参构造函数
	Neural_Network(const vector<vector<Intype>> &f,const vector<Outtype> &r);

	//获取参数列表信息
	void GetDataInfo(const bool showdata);

	//参数处理函数-1
	//功能:将空位用均值补齐
	void FillByMean();

	//参数处理函数-2
	//功能:将空位用中位数补齐
	void FillByMiddle();

	//参数处理函数-3
	//功能:数据的Z-score标准化
	void StdZscore();

	//参数处理函数-3
	//功能:归一化处理

	//打印特征均值数组
	void PrintFeatureMean();

	//打印特征中位数数组
	void PrintFeatureMiddle();

	//打印特征方差数组
	void PrintFeatureSigma();

	//打印结果数组的均值
	void PrintResultMean();

	//打印结果数组的中位数
	void PrintResultMiddle();

	//打印结果数组的方差
	void PrintResultSigma();

	//获取统计结果
	void PrintStatistics();

	//获取处理(或者没有处理)的特征列表
	vector<vector<double>> GetFeature();

	//获取处理(或者没有处理)的结果列表
	vector<double> GetResult();
};


//有参构造函数:特征集和结果集都是多维度
template<typename Intype, typename Outtype>
Neural_Network<Intype, Outtype>::Neural_Network(const vector<vector<Intype>> &f,const vector<Outtype> &r)
{
	feature_num = f.size();
	data_num = r.size();
	for (UI i = 0; i < f.size(); i++)
	{
		vector<double>change_arr;
		for (typename vector<Intype>::const_iterator it = f[i].begin(); it != f[i].end(); it++)
			change_arr.push_back(double(*it));
		feature.push_back(change_arr);
		mean_f.push_back(GetMeanOfVector(f[i]));
		middle_f.push_back(GetMiddleOfVector(f[i]));
		sigma_f.push_back(GetSigmaOfVector(f[i]));
		if (f[i].size() > data_num)
			data_num = f[i].size();
	}
	for (UI j = 0; j < r.size(); j++)
	{
		result.push_back(double(r[j]));
	}
	if (data_num < r.size())
		data_num = r.size();
	mean_r = GetMeanOfVector(r);
	middle_r = GetMiddleOfVector(r);
	sigma_r = GetSigmaOfVector(r);
}

//显示数据内容的工具
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::GetDataInfo(const bool showdata)
{
	cout << endl;
	cout << "data size : " << endl;
	cout << "feature : " << "[ " << feature_num << " x " << data_num << " ]" << endl;
	cout << "result  : " << "[ 1 x " << data_num << " ]" << endl;
	cout << endl;
	if (showdata)
	{
		cout << setiosflags(ios::left) << setw(8 * feature_num) << "feature : " << "result : " << endl;
		for (UI i = 0; i < data_num; i++)
		{
			for (UI j = 0; j < feature_num; j++)
			{
				if (i < feature[j].size())
				{
					cout << setw(8) << setprecision(2) << fixed << feature[j][i];
				}
				else
				{
					cout << setw(8) << " ";
				}
			}
			if (i < result.size())
			{
				cout << setw(8) << setprecision(2) << fixed << result[i];
			}
			cout << endl;
		}
	}
	cout << endl;
}

//用均值填充空缺的数据
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::FillByMean()
{
	for (UI i = 0; i < feature_num; i++)
	{
		UI lost = data_num - feature[i].size();
		if (lost > 0)
		{
			while (lost > 0)
			{
				feature[i].push_back(mean_f[i]);
				lost--;
			}
			mean_f[i] = GetMeanOfVector(feature[i]);
			middle_f[i] = GetMiddleOfVector(feature[i]);
			sigma_f[i] = GetSigmaOfVector(feature[i]);
		}
	}
	UI rlost = data_num - result.size();
	if (rlost > 0)
	{
		while (rlost > 0)
		{
			result.push_back(mean_r);
			rlost--;
		}
		mean_r = GetMeanOfVector(result);
		middle_r = GetMiddleOfVector(result);
		sigma_r = GetSigmaOfVector(result);
	}
}

//用中位数填充空缺的数据
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::FillByMiddle()
{
	for (UI i = 0; i < feature_num; i++)
	{
		UI lost = data_num - feature[i].size();
		if (lost > 0)
		{
			while (lost > 0)
			{
				feature[i].push_back(middle_f[i]);
				lost--;
			}
			mean_f[i] = GetMeanOfVector(feature[i]);
			middle_f[i] = GetMiddleOfVector(feature[i]);
			sigma_f[i] = GetSigmaOfVector(feature[i]);
		}
	}
	UI rlost = data_num - result.size();
	if (rlost > 0)
	{
		while (rlost > 0)
		{
			result.push_back(middle_r);
			rlost--;
		}
		mean_r = GetMeanOfVector(result);
		middle_r = GetMiddleOfVector(result);
		sigma_r = GetSigmaOfVector(result);
	}
}

//数据Z-score标准化
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::StdZscore()
{
	for (UI i = 0; i < feature.size(); i++)
	{
		for (UI j = 0; j < feature[i].size(); j++)
		{
			feature[i][j] = (feature[i][j] - mean_f[i]) / sigma_f[i];
		}
		mean_f[i] = GetMeanOfVector(feature[i]);
		middle_f[i] = GetMiddleOfVector(feature[i]);
		sigma_f[i] = GetSigmaOfVector(feature[i]);
	}
	for (UI i = 0; i < result.size(); i++)
	{
		result[i] = (result[i] - mean_r) / sigma_r;
	}
	mean_r = GetMeanOfVector(result);
	middle_r = GetMiddleOfVector(result);
	sigma_r = GetSigmaOfVector(result);
}

//显示特征值的均值数组
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::PrintFeatureMean()
{
	cout << "means of feature : " << endl;
	PrintVector(mean_f);
}

//显示特征数组中位数数组
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::PrintFeatureMiddle()
{
	cout << "middles of feature : " << endl;
	PrintVector(middle_f);
}

//显示特征数组方差数组
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::PrintFeatureSigma()
{
	cout << "sigmas of feature : " << endl;
	PrintVector(sigma_f);
}

//显示结果数组的均值
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::PrintResultMean()
{
	cout << "mean of result : " << endl;
	cout << setprecision(2) << fixed << mean_r << endl;
}

//显示结果数组的中位数
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::PrintResultMiddle()
{
	cout << "middle of result : " << endl;
	cout << setprecision(2) << fixed << middle_r << endl;
}

//显示结果数组的方差
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::PrintResultSigma()
{
	cout << "sigma of result : " << endl;
	cout << setprecision(2) << sigma_r << endl;
}

//获取统计结果
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::PrintStatistics()
{
	PrintFeatureMean();
	PrintFeatureMiddle();
	PrintFeatureSigma();
	PrintResultMean();
	PrintResultMiddle();
	PrintResultSigma();
}

//获取特征列表
template<typename Intype, typename Outtype>
vector<vector<double>>  Neural_Network<Intype, Outtype>::GetFeature()
{
	return feature;
}

//获取结果列表
template<typename Intype, typename Outtype>
vector<double>  Neural_Network<Intype, Outtype>::GetResult()
{
	return result;
}
