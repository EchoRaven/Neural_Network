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
void PrintVector(const vector<type>&arr)
{
	for (typename vector<type>::const_iterator it = arr.begin(); it != arr.end(); it++)
	{
		cout << setiosflags(ios::left) << setw(8) << fixed << setprecision(2) << *it;
	}
	cout << endl;
}

//Ѱ�����ֵ
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

//Ѱ����Сֵ
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

	//������ǰ��ת������
	vector<double> front_change;

	//������Ĳ�����;
	double compensate;

	//ת�����Ԥ�����
	vector<double> predict_result;

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
	void StdZscore(bool change_result);

	//����������-4
	//����:������һ������
	void Normalize();

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

	//��ӡԤ������ʵ�ʽ��
	void PrintPredict_Result();

	//��ȡ����(����û�д���)�������б�
	vector<vector<double>> GetFeature();

	//��ȡ����(����û�д���)�Ľ���б�
	vector<double> GetResult();

	//Ԥ�⺯��
	double Predict(const vector<Intype>&Inf);

	//��ʧ���ƺ���Loss
	double Loss();

	//�ݶ��½�����(front_change[index]���ݶ�)
	double Gradient_W(int index);

	//�ݶ��½�����(compensate���ݶ�)
	double Gradient_B();

	//����Ԥ������
	void Update_predict();

	//ѵ��ǰ��ת������(max_iter������������,eta��ÿ�α仯�Ĳ���)
	void TrainTransform(int max_iter, double eta);

	//��ȡת������
	void PrintTransform();
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
	for (UI k = 0; k < data_num; k++)
	{
		front_change.push_back(0);
		predict_result.push_back(0);
	}
	compensate = 0;
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
	Update_predict();
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
	Update_predict();
}

//����Z-score��׼��
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

//������һ������
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

//��ӡԤ������ʵ�ʽ��
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::PrintPredict_Result()
{
	cout << setiosflags(ios::left) << setw(8) << "Predict : " << setw(8) << "Result : " << endl;
	for (int i = 0; i < data_num; i++)
	{
		cout << setw(8) << predict_result[i] << setw(8) << result[i] << endl;
	}
}

//��ȡ�����б�
template<typename Intype, typename Outtype>
vector<vector<double>>  Neural_Network<Intype, Outtype>::GetFeature()
{
	return feature;
}

//��ȡ����б�
template<typename Intype, typename Outtype>
vector<double> Neural_Network<Intype, Outtype>::GetResult()
{
	return result;
}

//Ԥ������
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

//��ʧ���ƺ���
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

//�ݶ��½�����(����front_change[index]���ݶ�)
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

//�ݶ��½�����(����compensate���ݶ�)
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

//����Ԥ������
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

//ѵ��ת������
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::TrainTransform(int max_iter,double eta)
{
	for (int i = 0; i < max_iter; i++)
	{
		//��ÿһ�ε��������У������еĲ��������޸�
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

//��ȡת������
template<typename Intype, typename Outtype>
void Neural_Network<Intype, Outtype>::PrintTransform()
{
	cout << "ǰ��ת������Ϊ : ";
	for (UI i = 0; i < feature_num; i++)
	{
		cout << setiosflags(ios::left) << setw(8) << front_change[i];
	}
	cout << endl;
	cout << "��������Ϊ : ";
	cout << setiosflags(ios::left) << setw(8) << compensate << endl;
}
