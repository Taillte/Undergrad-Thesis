
class Grid
{
    private:
        int n_;
        int q_;
        double beta_;
        std::vector<int> data_;
        
        int index_(int x,int y) const
        {
            x = (x%n_ + n_)%n_;
            y = (y%n_ + n_)%n_;
            return x + n_*y;
        }
            
        
    public:
        Grid(int n,int q,double beta) : n_(n), q_(q),beta_(beta), data_((n)*(n))
        {
            for (int i=0;i<n*n;i++) data_[i] = 0.0;
        }
        int& operator() (int x,int y)
        {
            return data_[index_(x,y)];
        }
    
        int operator() (int x, int y) const
        {
            return data_[index_(x,y)];
        }
    
        int size() const {return n_;}
        int type() const {return q_;}
        double temp() const {return beta_;}
    
};
