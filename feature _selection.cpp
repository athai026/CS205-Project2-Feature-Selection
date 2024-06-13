#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <set>
#include <cmath>
#include "time.h"
#include <iomanip>

using namespace std;

void forward_selection(vector<vector<float>>&, int, int);
void backward_elimination(vector<vector<float>>&, int, int);
float LOO_cross_validation(vector<vector<float>>&, int, int, set<int>);
pair<set<int>, float> find_max_accuracy(vector<pair<set<int>, float>>&);
void default_rate(vector<vector<float>>&, int);
void print_set(set<int>);

int main () {
    string testFile = "";
    int choice;
    ifstream inFS;

    cout << "Welcome to Alice Thai's Feature Selection Algorithm." << endl;
    cout << "Type the name of the file to test: ";
    cin >> testFile;
    cout << endl;

    inFS.open(testFile);
    if (!inFS.is_open()) {
        cout << "Could not open " << testFile << endl;
        return 1;
    }

    clock_t start, end;
    start = clock(); //begin "timer"

    istringstream inSS;
    string line, col;
    int numRows = 0;
    int numCols = 0;

    float dataNum;
    vector<vector<float>> data;
    vector<float> temp;
    while (getline(inFS, line)) {
        temp.clear();
        inSS.clear();
        inSS.str(line);
        while (inSS >> col) {
            dataNum = stof(col);
            temp.push_back(dataNum);
        }
        data.push_back(temp);     
    } //read data into 2D vector

    numRows = data.size();
    numCols = data.at(0).size();
    cout << "This dataset has " << numCols - 1 << " features (not including the class attribute), with " << numRows << " instances." << endl;

    cout << "Type the number of the algorithm you want to run:" << endl;
    cout << "\t1) Forward Selection" << endl;
    cout << "\t2) Backward Elimination" << endl;
    cin >> choice;

    if (choice == 1) {
        forward_selection(data, numCols - 1, numRows);
    }
    else if (choice == 2) {
        backward_elimination(data, numCols - 1, numRows);
    }
    else {
        cout << "Invalid Choice." << endl;
    }

    inFS.close();

    end = clock(); //end "timer"
    float time_elapsed = float(end - start) / float(CLOCKS_PER_SEC); //divide by CLOCKS_PER_SEC to convert into seconds

    cout <<"Time elapsed: " << fixed << time_elapsed << setprecision(5) << " secs" << endl;

    return 0;
}

void forward_selection(vector<vector<float>>& data, int numFeatures, int numRows) {
    set<int> added_feature; //feature set to be tested
    pair<set<int>, float> feature_set; //feature set + accuracy
    vector<pair<set<int>, float>> possible_sets; //vector of feature_sets at level i to compare
    set<int> best_features = {}; //best feature set at level i to be passed onto level i+1
    vector<pair<set<int>, float>> best_sets; //vector of best feature sets from each level

    //find accuracy of all features
    for (int i = 1; i <= numFeatures; ++i) {
        best_features.insert(i);
    }
    feature_set = make_pair(best_features, LOO_cross_validation(data, numFeatures, numRows, best_features));
    possible_sets.push_back(feature_set);

    pair<set<int>, float> best_feature_to_add = find_max_accuracy(possible_sets);
    cout << "Running nearest neighbor with all " << numFeatures << " features, using \"leaving-one-out\" evaluation, I get an accuracy of " << best_feature_to_add.second * 100 << "%" << endl;
    
    default_rate(data, numRows); //default rate & empty feature set{}

    possible_sets.clear();
    best_features.clear();
    cout << "Beginning search." << endl << endl;
    for (int i = 1; i <= numFeatures; ++i) { //for each level
        for (int j = 1; j <= numFeatures; ++j) { //for each feature
            added_feature.clear();
            added_feature = best_features; //best feature set from previous level
            if (!added_feature.count(j)) { //if feature not already in set
                added_feature.insert(j); //add that feature
                feature_set = make_pair(added_feature, LOO_cross_validation(data, numFeatures, numRows, added_feature)); //find accuracy of added feature j
                possible_sets.push_back(feature_set); //add to list of all possible feature set combinations at level i
            }
        }
    
        for (int i = 0; i < possible_sets.size(); ++i) {
            cout << "\tUsing feature(s) {";
            print_set(possible_sets.at(i).first);
            cout << "} accuracy is " << possible_sets.at(i).second * 100 << "%" << endl;
        } //print accuracy of each feature set combination

        best_feature_to_add = find_max_accuracy(possible_sets); //find highest accuracy at level i
        if (!best_sets.empty()) {
            if (best_feature_to_add.second < best_sets.back().second) {
                cout << endl << "(Warning, Accuracy has decreased! Continuing search in case of local maxima)";
            } //warning if accuracy is decreasing
        }
        cout << endl << "Feature set {";
        print_set(best_feature_to_add.first);
        cout << "} was best, accuracy is " << best_feature_to_add.second * 100 << "%" << endl << endl;

        best_sets.push_back(best_feature_to_add); //add best feature set at level i to vector
        best_features = best_feature_to_add.first; //remember best feature set for next level
        possible_sets.clear();
    }

    best_feature_to_add = find_max_accuracy(best_sets); //find highest accuracy from all the levels = optimal feature set
    cout << endl << "Finished search!! The best feature subset is {";
    print_set(best_feature_to_add.first);
    cout << "}, which has an accuracy of " << best_feature_to_add.second * 100 << "%" << endl;
}

void backward_elimination(vector<vector<float>>& data, int numFeatures, int numRows) {
    set<int> added_feature; //feature set to be tested
    pair<set<int>, float> feature_set; //feature set + accuracy
    vector<pair<set<int>, float>> possible_sets; //vector of feature_sets at level i to compare
    set<int> best_features = {}; //best feature set at level i to be passed onto level i+1
    vector<pair<set<int>, float>> best_sets; //vector of best feature sets from  each level

    //start with full set of all features
    for (int i = 1; i <= numFeatures; ++i) {
        best_features.insert(i);
    }
    feature_set = make_pair(best_features, LOO_cross_validation(data, numFeatures, numRows, best_features));
    possible_sets.push_back(feature_set);

    pair<set<int>, float> best_feature_to_add = find_max_accuracy(possible_sets);
    cout << "Running nearest neighbor with all " << numFeatures << " features, using \"leaving-one-out\" evaluation, I get an accuracy of " << best_feature_to_add.second * 100 << "%" << endl;
    cout << "Beginning search." << endl << endl;

    cout << "Feature set {";
    print_set(best_feature_to_add.first);
    cout << "} was best, accuracy is " << best_feature_to_add.second * 100 << "%" << endl << endl;

    possible_sets.clear();
    best_sets.push_back(feature_set); 
    for (int i = 1; i < numFeatures; ++i) { //for each level
        for (int j = 1; j <= numFeatures; ++j) { //for each feature
            added_feature = best_features; //best feature set from previous level
            if (added_feature.count(j)) { //if feature already in set
                added_feature.erase(j); //remove from set
                feature_set = make_pair(added_feature, LOO_cross_validation(data, numFeatures, numRows, added_feature)); //find accuracy without feature j
                possible_sets.push_back(feature_set); //add to list of all possible feature set combinations at level i
            }
        }

        for (int i = 0; i < possible_sets.size(); ++i) {
            cout << "\tUsing feature(s) {";
            print_set(possible_sets.at(i).first);
            cout << "} accuracy is " << possible_sets.at(i).second * 100 << "%" << endl;
        } //print accuracy of each feature set combination

        best_feature_to_add = find_max_accuracy(possible_sets); //find highest accuracy at level i
        if (!best_sets.empty()) {
            if (best_feature_to_add.second < best_sets.back().second) {
                cout << endl << "(Warning, Accuracy has decreased! Continuing search in case of local maxima)";
            } //warning if accuracy is decreasing
        }
        cout << endl << "Feature set {";
        print_set(best_feature_to_add.first);
        cout << "} was best, accuracy is " << best_feature_to_add.second * 100 << "%" << endl << endl;

        best_sets.push_back(best_feature_to_add);
        best_features = best_feature_to_add.first;
        possible_sets.clear();
    }

    default_rate(data, numRows); //default rate

    best_feature_to_add = find_max_accuracy(best_sets); //find highest accuracy from all the levels = optimal feature set
    cout << endl << "Finished search!! The best feature subset is {";
    print_set(best_feature_to_add.first);
    cout << "}, which has an accuracy of " << best_feature_to_add.second * 100 << "%" << endl;

}

float LOO_cross_validation(vector<vector<float>>& data, int numFeatures, int numRows, set<int> features) {
    set<int>::iterator itr;
    vector<int> featuresTest;
    for(itr = features.begin(); itr != features.end(); itr++) {
        featuresTest.push_back(*itr);
    }

    //find euclidean distance
    vector<float> distance;
    float sum_distances = 0;
    float euclidean = 0;
    float minDist = INFINITY;
    int minIndex = -1;
    float numCorrect = 0;
    bool match = false;
    for (int i = 0; i < numRows; ++i) {
        sum_distances = 0;
        minDist = INFINITY;
        minIndex = -1;
        for (int j = 0; j < numRows; ++j) {
            if (i != j) {
                sum_distances = 0;
                for (int k = 0; k < featuresTest.size(); ++k) {
                    sum_distances += pow(data.at(i).at(featuresTest.at(k)) - data.at(j).at(featuresTest.at(k)), 2);
                }
                euclidean = sqrt(sum_distances);
                if (euclidean < minDist) {
                    minDist = euclidean;
                    minIndex = j;
                }
            }
        }
        if (data.at(i).at(0) == data.at(minIndex).at(0)) { //if classes match, increment numCorrect
            ++numCorrect;
        }
    }
    float accuracy = numCorrect / numRows;
    return accuracy;
}

pair<set<int>, float> find_max_accuracy(vector<pair<set<int>, float>>& possible_sets) {
    int maxIndex = -1;
    float bestAccuracy = 0;

    for (int i = 0; i < possible_sets.size(); ++i) {
        if (bestAccuracy < possible_sets.at(i).second) {
            bestAccuracy = possible_sets.at(i).second;
            maxIndex = i;
        }
    }

    return possible_sets.at(maxIndex);
}

void default_rate(vector<vector<float>>& data, int numRows) {
    float cat1 = 0;
    float cat2 = 0;
    for (int i = 0; i < numRows; ++i) {
        if (data.at(i).at(0) == 1) {
            ++cat1;
        }
        else {
            ++cat2;
        }
    }
    cout << "The default rate using zero features is ";
    if (cat1 > cat2) {
        cout  << cat1 / numRows * 100 << "%" << endl;
    }
    else {
        cout << cat2 / numRows * 100 << "%" << endl;
    } 
}

void print_set(set<int> toPrint) {
    set<int>::iterator itr;

    for (itr = toPrint.begin(); itr != toPrint.end(); itr++) {
        if (itr != toPrint.begin()) {
            cout << ",";
        }
        cout << *itr;
    }
}