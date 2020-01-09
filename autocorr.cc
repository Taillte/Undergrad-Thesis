#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

#include "Grid.h"



double deltaE(const Grid& phi,int x,int y,int change)
{
    double sum = 0.0;
    sum+= (1-cos(phi(x,y)*2*M_PI/double(phi.type()) - phi(x+1,y)*2*M_PI/double(phi.type())));
    sum+= (1-cos(phi(x,y)*2*M_PI/double(phi.type()) - phi(x-1,y)*2*M_PI/double(phi.type())));
    sum+= (1-cos(phi(x,y)*2*M_PI/double(phi.type()) - phi(x,y+1)*2*M_PI/double(phi.type())));
    sum+= (1-cos(phi(x,y)*2*M_PI/double(phi.type()) - phi(x,y-1)*2*M_PI/double(phi.type())));
    
    double newsum = 0.0;
    newsum+= (1-cos(change*2*M_PI/double(phi.type()) - phi(x+1,y)*2*M_PI/double(phi.type())));
    newsum+= (1-cos(change*2*M_PI/double(phi.type()) - phi(x-1,y)*2*M_PI/double(phi.type())));
    newsum+= (1-cos(change*2*M_PI/double(phi.type()) - phi(x,y+1)*2*M_PI/double(phi.type())));
    newsum+= (1-cos(change*2*M_PI/double(phi.type()) - phi(x,y-1)*2*M_PI/double(phi.type())));
    
    return newsum - sum;

}

void update(Grid& phi)
{
    int x = (rand()/double(RAND_MAX))*phi.size();
    int y = (rand()/double(RAND_MAX))*phi.size();
    int change = (rand()/double(RAND_MAX))*phi.type();

    double Denergy = deltaE(phi,x,y,change);
    //cout<<"energy change = "<<Denergy<<"\n";
    
    if(Denergy<=0)
    {
        phi(x,y)=change;
        //cout<<"goes to "<<phi(x,y)<<"\n";
    }
    else
    {
        double prob = exp(-phi.temp()*Denergy);
        //cout<<"could change \n";
        if(rand()/double(RAND_MAX) <= prob)
        {
            phi(x,y)=change;
            //cout<<"change \n";
        }
    }

}

double magnetization(const Grid& phi)
{
    std::vector<int> quantity(phi.type());
    for(int i=0;i<phi.size();i++)
    {
        for(int j=0;j<phi.size();j++)
        {
            quantity[phi(i,j)]+=1;
        }
    }
    int max_index = 0;
    int max_q = 0;
    for(int n=0;n<phi.type();n++)
    {
        if(quantity[n]>max_q)
        {
            max_q=quantity[n];
            max_index=n;
        }
    }
    double m = (1.0/(phi.size()*phi.size()) * max_q);
    return (1.0/(phi.type() - 1))*(phi.type()*m -1);
}

void initialise(Grid& phi)
{
    for(int i=0;i<phi.size();i++)
    {
        for(int j=0;j<phi.size();j++)
        {
            phi(i,j) = int((rand()/double(RAND_MAX))*phi.type());
        }
    }
}

            
int main()
{
    srand(time(NULL));
    int max_binsize = 100000;
    double beta = 1.0/2.3;
    int q = 2;
    double number_samples = 10000000;

    
    vector<double> magnetizations(number_samples);
    
    Grid phi(20,q,beta);
    initialise(phi);
    
    for(int n=0;n<10000000;n++)
    {
        update(phi);
    }
    
    for(int i=0;i<number_samples;i++)
    {
        update(phi);
        magnetizations[i] = magnetization(phi);
    }
    
    for(int binsize = 1; binsize<max_binsize; binsize=binsize+50)
    {
        int num_bins = number_samples/binsize;

        vector<double> mag(num_bins);
        
        for(double bin=0;bin<num_bins;bin+=1)
        {
            double sum = 0.0;

            for(int m=bin*binsize;m<(bin+1)*binsize;m++)
            {
                sum+=magnetizations[m];
            }
            
            sum=sum/binsize;
            mag[bin]=sum;
        }
        double mean_mag = 0.0;
        for(int i=0;i<num_bins;i++)
        {
            mean_mag+=mag[i];
        }
        mean_mag = mean_mag/num_bins;
        double st_dev_mag = 0.0;
        for(int i=0;i<num_bins;i++)
        {
            st_dev_mag+=(mag[i]-mean_mag)*(mag[i]-mean_mag);
        }
        st_dev_mag= sqrt(st_dev_mag/((num_bins-1)*num_bins));
            
        cout<<binsize<<" "<<mean_mag<<" "<<st_dev_mag<<"\n";
    }
    

}
