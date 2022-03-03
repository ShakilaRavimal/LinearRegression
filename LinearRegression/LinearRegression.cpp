#include <iostream>
#include <vector>
#include <fstream>
#include "bits-stdc++.h"

using namespace std;

vector<string> split(const string& str, const string& delim)
{
    vector<string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == string::npos)
        {
            pos = str.length();
        }
        string token = str.substr(prev, pos - prev);
        if (!token.empty())
        {
            tokens.push_back(token);
        }
        prev = pos + delim.length();
    } while (pos < str.length() && prev < str.length());

    return tokens;
}

template<typename T>
void column_drop(vector<int> drops, vector<vector<T>>& tdata)
{
    sort(drops.begin(), drops.end());
    for (int k = 0; k < drops.size(); k++)
    {
        if (k > 0)
        {
            drops[k] = drops[k] - 1;
        }
        for (int i = 0; i < tdata.size(); i++)
        {
            tdata[i].erase(tdata[i].begin() + drops[k]);
        }

    }
}

vector<vector<double>> readprepareTraindataset(const char* fname)
{
    vector<string> data;
    vector<vector<double>> tdata;
    ifstream file(fname);
    string line = "";
    int u = 0;
    while (getline(file, line))
    {
        if (line != "")
        {
            for (int i = 0; i + 1 < line.length(); i++)
            {
                if (line[i] == ',' && line[i + 1] == ',')
                {
                    line.insert(i + 1, "0");
                }
            }
            data.push_back(line);
            u++;
        }

    }

    file.close();

    tdata.resize(data.size() - 1);

    for (int i = 1; i < data.size(); i++)
    {
        vector<string> str = split(data[i], ",");

        for (int j = 0; j < str.size(); j++)
        {
            tdata[i - 1].push_back(atof(str[j].c_str()));

        }
    }

    vector<int> drops = { 0 };
    column_drop(drops, tdata);

    return tdata;

}


vector<double> initialize_para(vector<vector<double>>& tdata)
{
    vector<double> slopes;
    set<double> order;
    double Ymin = 0, Xmin = 0, Ymax = 0, Xmax = 0, Ydif = 0;

    for (int i = 0; i < tdata.size(); i++)
    {
        order.insert(tdata[i][tdata[i].size() - 1]);
    }

    Ymin = *next(order.begin(), 0);
    Ymax = *next(order.begin(), order.size() - 1);
    Ydif = Ymax - Ymin;
    order.clear();

    for (int j = 0; j < tdata[0].size() - 1; j++)
    {
        for (int i = 0; i < tdata.size(); i++)
        {
            order.insert(tdata[i][j]);
        }
        Xmin = *next(order.begin(), 0);
        Xmax = *next(order.begin(), order.size() - 1);
        slopes.push_back(Ydif / (Xmax - Xmin));
        order.clear();
    }

    return slopes;

}

double cal_SE(const double pre, const double tar)
{
    return pow((tar - pre), 2);
}

double predict_Y(const int i, vector<vector<double>>& tdata, vector<double>& slopes, const double intercept)
{
    double Y_bar = intercept;
    for (int j = 0; j < tdata[i].size() - 1; j++)
    {
        Y_bar += (slopes[j] * tdata[i][j]);
    }

    return Y_bar;
}

double Pdifferentiate_loss_wst_intercept(vector<vector<double>>& tdata, vector<double>& predictions)
{
    double dif = 0;
    for (int i = 0; i < tdata.size(); i++)
    {
        dif += (tdata[i][tdata[i].size() - 1] - predictions[i]);
    }

    return -((2 * dif) / (tdata.size() - 1));
}

double Pdifferentiate_loss_wst_slope(const int j, vector<vector<double>>& tdata, vector<double>& predictions)
{
    double dif = 0;
    for (int i = 0; i < tdata.size(); i++)
    {
        dif += (tdata[i][j] * (tdata[i][tdata[i].size() - 1] - predictions[i]));

    }

    return -((2 * dif) / (tdata.size() - 1));
}

void Gradient_Descend(vector<vector<double>>& tdata, vector<double>& predictions, vector<double>& slopes, double& intercept, const double learning_rate)
{
    intercept = intercept - (learning_rate * Pdifferentiate_loss_wst_intercept(tdata, predictions));

    for (int j = 0; j < slopes.size(); j++)
    {
        slopes[j] = slopes[j] - (learning_rate * Pdifferentiate_loss_wst_slope(j, tdata, predictions));

    }

}

double mean_squard_error(vector<vector<double>>& tdata, vector<double>& slopes, double& intercept, vector<double>& predictions)
{
    double MeanSE = 0;
    for (int i = 0; i < tdata.size(); i++)
    {
        double predicted = predict_Y(i, tdata, slopes, intercept);
        MeanSE += cal_SE(predicted, tdata[i][tdata[i].size() - 1]);
        predictions[i] = predicted;

    }

    return MeanSE / (2 * (tdata.size() - 1));
}

void Linear_regression(vector<vector<double>>& tdata, const double learning_rate, const double precision, vector<double>& slopes, double& intercept)
{
    double loss = 2;
    vector<double> predictions;

    predictions.resize(tdata.size());
    slopes = initialize_para(tdata);
    cout << "loss--> " << mean_squard_error(tdata, slopes, intercept, predictions) << endl;

    while (loss > precision)
    {
        Gradient_Descend(tdata, predictions, slopes, intercept, learning_rate);
        loss = mean_squard_error(tdata, slopes, intercept, predictions);
        cout << "loss--> " << loss << endl;

    }
}

int main()
{
    vector<double> slopes;
    double intercept;
    vector<vector<double>> tdata = readprepareTraindataset("Advertising.csv");
    Linear_regression(tdata, 0.000005, 1.4, slopes, intercept);

    cout << "--------Predictions-------" << endl;

    vector<vector<double>> test_data = { {230.1,37.8,69.2,22.1} };

    cout << predict_Y(0, test_data, slopes, intercept) << endl;

    return 0;
}
