//2151130 ͯ����

#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#include <iomanip>
#include <algorithm>
#define UI unsigned int
using namespace std;

//ͨ�ú���
//��ȡ����ƽ��ֵ(��ʲô���͵���ֵ�ͷ���ʲô���͵���ֵ)
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

//��ȡ�������λ��
template<typename type>
double GetMiddleOfVector(const vector<type>&arr)
{
	vector<type>temp(arr);
	sort(temp.begin(), temp.end());
	if (temp.size() % 2 == 0)
		return double((temp[temp.size() / 2] + temp[temp.size() / 2 - 1])) / 2;
	return double(temp[temp.size() / 2]);
}

//��ȡ����ķ���
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

//��ӡ����
template<typename type>
void PrintVector(vector<type>arr)
{
	for (typename vector<type>::iterator it = arr.begin(); it != arr.end(); it++)
	{
		cout << setiosflags(ios::left) << setw(8) << fixed << setprecision(2) << *it;
	}
	cout << endl;
}


//ģ����ʵ��
template<typename Intype, typename Outtype>
class Neural_Network
{
private:
	//����������˲���
	vector<vector<double>> feature;

	//���Ӳ�����Ϣ
	//�б��ֵ
	vector<double> mean_f;

	//�б���λ��
	vector<double> middle_f;

	//�б�ķ���
	vector<double>sigma_f;

	//����������˲���
	vector<double> result;

	//���Ӳ�����Ϣ
	//�б��ֵ
	double mean_r;

	//�б���λ��
	double middle_r;

	//�б���
	double sigma_r;

	//������ά��
	UI feature_num;

	//��������
	UI data_num;

public:
	//Ĭ�Ϲ��캯��
	Neural_Network() {};

	//�вι��캯��
	Neural_Network(const vector<vector<Intype>> &f,const vector<Outtype> &r);

	//��ȡ�����б���Ϣ
	void GetDataInfo(const bool showdata);

	//����������-1
	//����:����λ�þ�ֵ����
	void FillByMean();

	//����������-2
	//����:����λ����λ������
	void FillByMiddle();

	//����������-3
	//����:���ݵ�Z-score��׼��
	void StdZscore();

	//����������-3
	//����:��һ������

	//��ӡ������ֵ����
	void PrintFeatureMean();

	//��ӡ������λ������
	void PrintFeatureMiddle();

	//��ӡ������������
	void PrintFeatureSigma();

	//��ӡ�������ľ�ֵ
	void PrintResultMean();

	//��ӡ����������λ��
	void PrintResultMiddle();

	//��ӡ�������ķ���
	void PrintResultSigma();

	//��ȡͳ�ƽ��
	void PrintStatistics();

	//��ȡ����(����û�д���)�������б�
	vector<vector<double>> GetFeature();

	//��ȡ����(����û�д���)�Ľ���б�
	vector<double> GetResult();
};


//�вι��캯��:�������ͽ�������Ƕ�ά��
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

//��ʾ�������ݵĹ���
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

//�þ�ֵ����ȱ������
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

//����λ������ȱ������
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

//����Z-score��׼��
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

//��ʾ����ֵ�ľ�ֵ����
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::PrintFeatureMean()
{
	cout << "means of feature : " << endl;
	PrintVector(mean_f);
}

//��ʾ����������λ������
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::PrintFeatureMiddle()
{
	cout << "middles of feature : " << endl;
	PrintVector(middle_f);
}

//��ʾ�������鷽������
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::PrintFeatureSigma()
{
	cout << "sigmas of feature : " << endl;
	PrintVector(sigma_f);
}

//��ʾ�������ľ�ֵ
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::PrintResultMean()
{
	cout << "mean of result : " << endl;
	cout << setprecision(2) << fixed << mean_r << endl;
}

//��ʾ����������λ��
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::PrintResultMiddle()
{
	cout << "middle of result : " << endl;
	cout << setprecision(2) << fixed << middle_r << endl;
}

//��ʾ�������ķ���
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::PrintResultSigma()
{
	cout << "sigma of result : " << endl;
	cout << setprecision(2) << sigma_r << endl;
}

//��ȡͳ�ƽ��
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

//��ȡ�����б�
template<typename Intype, typename Outtype>
vector<vector<double>>  Neural_Network<Intype, Outtype>::GetFeature()
{
	return feature;
}

//��ȡ����б�
template<typename Intype, typename Outtype>
vector<double>  Neural_Network<Intype, Outtype>::GetResult()
{
	return result;
}
