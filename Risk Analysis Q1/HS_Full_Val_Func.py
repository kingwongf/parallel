def HS_Bootstrap_Full_Val_w_Opt(alpha, epsilon, n, B, port_weights, CVar_weights):

    # matrix of the 6 stocks returns of 1536 days
    d_ret_tup = np.transpose(np.array([VZ_log_ret_close,INTC_log_ret_close, JPM_log_ret_close , AAPL_log_ret_close,  MSFT_log_ret_close, PG_log_ret_close]))

    # repeat bootstrap for B times to generate B many simulated VaR and ES
    HS_full_B_list = []
    HS_full_ES_B_list = []

    B = 10000

#### run Bootstrap here parallerised ######
    np.mean(sim_HS_full_Val)
    np.mean(sim_HS_full_Val_ES)

    def Boostrap(B):

    # define parameters

        alpha = 0.95
        epsilon = 0.005
        n = 10
        # matrix of randomly sample B times of the six returns, n days

        T_n_matrix = d_ret_tup[np.random.choice(d_ret_tup.shape[0], size=[1536,n], replace=True)]

        # convert daily return to n-period returns

        sum_Tn = np.sum(T_n_matrix, axis = 1)

        # calculate bootstrap stock values by multiplying boststrap stock returns and last day stock value

        ten_day_port_value = np.multiply(np.exp(sum_Tn), last_day_port_v)


        # stock order in , 1. VZ    2. INTC     3. JPM      4. AAPL     5. MSFT     6. PG

        ###     initial option values on 2018-02-09      ####

        call_int_AAPL= 20.05    # short 7, 140 calls, Apr 20, implied vol = 0.3954, div yield = 0.052781
        put_int_MSFT = 12.225   # long 6, 95 puts, Jul 20, implied vol = 0.4103, div yield = 0.018
        call_int_PG = 1.695      # long 10, 85 calls, Jun 16, implied vol = 0.2046, div yield = 0.0344

        opt_int = np.array([call_int_AAPL, put_int_MSFT, call_int_PG])


        ###     maturity of the options     ###
        call_aapl_days = date(2018,2,9) - date(2018,4,20)       # 70
        call_aapl_days = - call_aapl_days.days
        put_msft_days = date(2018,2,9) - date(2018,7,20)        # 161
        put_msft_days = - put_msft_days.days
        call_pg_days = date(2018,2,9) - date(2018,6,15)         # 126
        call_pg_days = - call_pg_days.days

        ### interpolate risk free rate of the options according to the maturity and the given US LIBOR term of structure ###
        rate_aapl = 0.0168324 + (call_aapl_days-60)*(0.0181050-0.0168324)/(90-60)
        rate_msft = 0.0181050 + (put_msft_days-90)*(0.0202633-.0181050)/(180-90)
        rate_pg = 0.0181050 + (call_pg_days-90)*(0.0202633-.0181050)/(180-90)

        # calculate AAPL calls, MSFT puts, PG calls, 10 day simulated prices based on simulated stock prices
        # the indexing [] indicates holding 0. call or 1. put

        aapl_opt = OptionPrice(ten_day_port_value[:,3], 140, 0.3954, 0.052781, rate_aapl, call_aapl_days/360)[0]
        msft_opt = OptionPrice(ten_day_port_value[:,4], 95, 0.4103, 0.018, rate_msft, put_msft_days/360)[1]
        pg_opt = OptionPrice(ten_day_port_value[:,5], 85, 0.2046, 0.0344, rate_pg, call_pg_days/360)[0]


        # Simulated Option portfolio value, 1. AAPL calls 2. MSFT puts 3. PG Calls
        opt_ten_val = np.array([aapl_opt, msft_opt, pg_opt])
        opt_ten_vel = np.transpose(opt_ten_val)

        # Simulated Full portfolio value by combining stocks and options prices
        ten_day_port_value_w_opt = np.concatenate((ten_day_port_value, opt_ten_vel), axis = 1)


        # multiply the ten day portfolio value by number of holdings/ weights
        weighted_ten_day_port_w_opt = np.multiply(ten_day_port_value_w_opt, port_weights)

        # Getting simulated portfolio returns, including option returns
        # First add initial option values to the list of securities prices on 2018-02-09
        last_day_port_w_opt = np.append(last_day_port_v, opt_int)

        weighted_last_day_port_w_opt = np.sum(np.multiply(last_day_port_w_opt, port_weights), axis =0)


        # Calculate simulated portfolio returns by dividing simulated prices by the security prices on 2018-02-09 respectively
        ten_day_sim_ret = np.log(np.divide(np.sum(weighted_ten_day_port_w_opt, axis = 1), weighted_last_day_port_w_opt))

        # plt.hist(ten_day_sim_ret) #TODO histogram of HS full Val
        # plt.show()

        # Calculate simulated total portfolio returns

        # Calculate VaR by locating the 1st percentile
        sim_HS_full_Val = -np.percentile(ten_day_sim_ret, (1 - alpha)*100, interpolation='linear')



        # Calculare Expected Shortfall by taking the mean of simulated returns which are lower than the VaR
        ES = ten_day_sim_ret[ten_day_sim_ret < -sim_HS_full_Val]
        sim_HS_full_Val_ES = -np.mean(ES)

        return sim_HS_full_Val, sim_HS_full_Val_ES, ten_day_sim_ret



    ############     Calculate Marginal VaR and Conditional VaR of HS Full Revaluation    #########


    ten_day_sim_port_ret_HS = ten_day_sim_ret   # simulated portfolio returns
    HS_full_Val_VaR = HS_full_Val               # calculated VaR
    ten_day_sim_sec_ret_HS = np.log(np.divide(ten_day_port_value_w_opt, last_day_port_w_opt)) # calculate simulated securities returns

    l = []  # create a new list for MVaR

    # Conditioning on portfolio returns between -VaR - epsilon and -VaR + epsilon, return list of conditioned simulated securities returns
    for i in range(0,len(ten_day_sim_port_ret_HS)):
        if (-HS_full_Val_VaR - epsilon) <= ten_day_sim_port_ret_HS[i] <= (-HS_full_Val_VaR + epsilon):
            l.append(ten_day_sim_sec_ret_HS[i])
    l = np.asarray(l)

    # the means of the conditioned securities returns are the MVaR of each security
    MVaR = -np.mean(l, axis = 0 )

    # cauclate the percentage weights of each security holdings
    sum_weights = np.sum(CVar_weights)
    percent_weights = np.divide(CVar_weights, sum_weights)

    # multiply MVaR by percentage weights returns CVaR
    CVaR = np.multiply(MVaR, percent_weights)

    # print out MVaR and CVaR of each security
    name = np.array(["VZ","INTC","JPM","AAPL","MSFT","PG","AAPL opt", "MSFT opt","PG opt"])
    MCVaR = {'MVaR':MVaR, 'CVaR':CVaR}

    # print(pd.DataFrame(data=MCVaR, index=name))       # TODO printing out MVaR and CVaR

    print(HS_full_Val_VaR, HS_full_Val_ES, MVaR, CVaR)

    return HS_full_Val_VaR, HS_full_Val_ES, MVaR, CVaR