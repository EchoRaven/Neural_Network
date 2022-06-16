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
void PrintVector(const vector<type>&arr)
{
	for (typename vector<type>::const_iterator it = arr.begin(); it != arr.end(); it++)
	{
		cout << setiosflags(ios::left) << setw(8) << fixed << setprecision(2) << *it;
	}
	cout << endl;
}

//寻找最大值
template<typename type>
type FindMax(const vector<type>&arr)
{
	type maxnum = arr[0];
	for (typename vector<type>::const_iterator it = arr.begin(); it != arr.end(); it++)
	{
		if (*it > maxnum)
			maxnum = *it;
	}
	return maxnum;
}

//寻找最小值
template<typename type>
type FindMin(const vector<type>&arr)
{
	type minnum = arr[0];
	for (typename vector<type>::const_iterator it = arr.begin(); it != arr.end(); it++)
	{
		if (*it < minnum)
			minnum = *it;
	}
	return minnum;
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

	//神经网络前向转化数组
	vector<double> front_change;

	//神经网络的补偿数;
	double compensate;

	//转化后的预估结果
	vector<double> predict_result;

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
	void StdZscore(bool change_result);

	//参数处理函数-4
	//功能:特征归一化处理
	void Normalize();

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

	//打印预测结果和实际结果
	void PrintPredict_Result();

	//获取处理(或者没有处理)的特征列表
	vector<vector<double>> GetFeature();

	//获取处理(或者没有处理)的结果列表
	vector<double> GetResult();

	//预测函数
	double Predict(const vector<Intype>&Inf);

	//损失估计函数Loss
	double Loss();

	//梯度下降函数(front_change[index]的梯度)
	double Gradient_W(int index);

	//梯度下降函数(compensate的梯度)
	double Gradient_B();

	//更新预测数组
	void Update_predict();

	//训练前向转化数组(max_iter是最大迭代次数,eta是每次变化的步长)
	void TrainTransform(int max_iter, double eta);

	//获取转化函数
	void PrintTransform();
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
	for (UI k = 0; k < data_num; k++)
	{
		front_change.push_back(0);
		predict_result.push_back(0);
	}
	compensate = 0;
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
	Update_predict();
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
	Update_predict();
}

//数据Z-score标准化
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::StdZscore(bool change_result)
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
	if (change_result)
	{
		for (UI i = 0; i < result.size(); i++)
		{
			result[i] = (result[i] - mean_r) / sigma_r;
		}
		mean_r = GetMeanOfVector(result);
		middle_r = GetMiddleOfVector(result);
		sigma_r = GetSigmaOfVector(result);
	}
	Update_predict();
}

//特征归一化处理
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::Normalize()
{
	double maxnum;
	double minnum;
	double deta;
	for (UI i = 0; i < feature.size(); i++)
	{
		maxnum = FindMax(feature[i]);
		minnum = FindMin(feature[i]);
		deta = maxnum - minnum;
		for (UI j = 0; j < feature[i].size(); j++)
		{
			feature[i][j] = (feature[i][j] - minnum) / deta;
		}
		mean_f[i] = GetMeanOfVector(feature[i]);
		middle_f[i] = GetMiddleOfVector(feature[i]);
		sigma_f[i] = GetSigmaOfVector(feature[i]);
	}
	Update_predict();
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

//打印预测结果与实际结果
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::PrintPredict_Result()
{
	cout << setiosflags(ios::left) << setw(8) << "Predict : " << setw(8) << "Result : " << endl;
	for (int i = 0; i < data_num; i++)
	{
		cout << setw(8) << predict_result[i] << setw(8) << result[i] << endl;
	}
}

//获取特征列表
template<typename Intype, typename Outtype>
vector<vector<double>>  Neural_Network<Intype, Outtype>::GetFeature()
{
	return feature;
}

//获取结果列表
template<typename Intype, typename Outtype>
vector<double> Neural_Network<Intype, Outtype>::GetResult()
{
	return result;
}

//预估函数
template<typename Intype, typename Outtype>
double Neural_Network<Intype, Outtype>::Predict(const vector<Intype>&Inf)
{
	double r = 0;
	for (UI i = 0; i < Inf.size(); i++)
	{
		r += double(Inf[i]) * double(front_change[i]);
	}
	r += compensate;
	return r;
}

//损失估计函数
template<typename Intype, typename Outtype>
double Neural_Network<Intype, Outtype>::Loss()
{
	Update_predict();
	double total = 0;
	UI Min = min(result.size(), predict_result.size());
	for (UI i = 0; i < Min; i++)
	{
		total += (predict_result[i] - result[i]) * (predict_result[i] - result[i]);
	}
	total = total / Min / 2;
	return total;
}

//梯度下降函数(计算front_change[index]的梯度)
template<typename Intype, typename Outtype>
double Neural_Network<Intype, Outtype>::Gradient_W(int index)
{
	double total = 0;
	UI Min = min(result.size(), predict_result.size());
	for (UI i = 0; i < Min; i++)
	{
		total += (predict_result[i] - result[i]) * feature[index][i];
	}
	total = total / Min;
	return total;
}

//梯度下降函数(计算compensate的梯度)
template<typename Intype, typename Outtype>
double Neural_Network<Intype, Outtype>::Gradient_B()
{
	double total = 0;
	UI Min = min(result.size(), predict_result.size());
	for (UI i = 0; i < Min; i++)
	{
		total += (predict_result[i] - result[i]);
	}
	total = total / Min;
	return total;
}

//更新预测数组
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::Update_predict()
{
	for (UI i = 0; i < data_num; i++)
	{
		vector<Intype>arr;
		for (UI j = 0; j < feature_num; j++)
			arr.push_back(feature[j][i]);
		predict_result[i] = Predict(arr);
	}
}

//训练转化函数
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::TrainTransform(int max_iter,double eta)
{
	for (int i = 0; i < max_iter; i++)
	{
		//在每一次迭代过程中，对所有的参数进行修改
		for (UI i = 0; i < feature_num; i++)
		{
			Update_predict();
			double G1 = abs(Gradient_W(i));
			double temp = front_change[i];
			front_change[i] = temp + eta;
			Update_predict();
			double G2= abs(Gradient_W(i));
			front_change[i] = temp - eta;
			Update_predict();
			double G3 = abs(Gradient_W(i));
			front_change[i] = temp;
			if (G2 < G1 && G2 < G3)
			{
				front_change[i] = temp + eta;
			}
			else if (G3 < G1 && G3 < G2)
			{
				front_change[i] = temp + eta;
			}
			Update_predict();
		}
		Update_predict();
		double G1 = abs(Gradient_B());
		double temp = compensate;
		compensate = temp + eta;
		Update_predict();
		double G2 = abs(Gradient_B());
		compensate = temp - eta;
		Update_predict();
		double G3 = abs(Gradient_B());
		compensate = temp;
		if (G2 < G1 && G2 < G3)
		{
			compensate = temp + eta;
		}
		else if (G3 < G1 && G3 < G2)
		{
			compensate = temp + eta;
		}
		Update_predict();
	}
	return;
}

//获取转化函数
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::PrintTransform()
{
	cout << "前向转化数组为 : ";
	for (UI i = 0; i < feature_num; i++)
	{
		cout << setiosflags(ios::left) << setw(8) << front_change[i];
	}
	cout << endl;
	cout << "补偿参数为 : ";
	cout << setiosflags(ios::left) << setw(8) << compensate << endl;
}
