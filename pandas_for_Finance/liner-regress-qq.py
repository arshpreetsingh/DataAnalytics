There are many ways we analyses stocks, As we are aware of the factor that Stocks are important thing to consider when we talk about Data and stuff I believe Logistic regression is the one we should consider to use in our life, there are many ways to use Logistic Regression on your system, I will go with very simple methods those will Explain in almost layman's terms when and how you are doing analysis.


Apan pehlan logistic regression wala code likhiye, and sara apply kariye 56% wale takk,,

fer apan cross validation karange

fer apan Data bias wala dekhang v

Supervised learning:

The training data consist of a set of training examples. In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal). A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples.

(xi,yi), i∈{1,...,N},

So following are our Training sets:

now xi is predictor representing the lagged stock market returns or volume traded.

 yi representing the associated response/observation variables (such as the stock market return today).

forecasting accuracy measures:

Mean-Squared Error, Mean Absolute Deviation and Root-Mean-Squared Error


Forecasting Factors:

Factors effecting forecasting predictoins....

    In this article we are going to restrict the factors to time lags of the current percentage returns. 

What are time lags and which other predictors should be considered?

Even simple machine learning techniques will produce good results on well-chosen factors. Note that the converse is not often the case. "Throwing an algorithm at a problem" will usually lead to poor forecasting accuracy.


"Just throwing an algorithm to problem is really not the job of Data - Scientist"

For this forecaster specifically, I have chosen the first and second time lags of the percentage returns as the predictors for the current stock market direction.

Statistical Tests:

although there are statistical tests available which can demonstrate the predictive capability of each factor.



Mera question eh hai ke how this hit rate is going to effect my decison, main eh kiven decide karna hai ke j eh hit rate +ve ja -ve hoya main ki karna hai and kiven karna hai??

esnu main apne trading model ch kiven implement karna hai..

Ethe dekho dhyaan naal, Predict kar lea hai -

    jehda apan compare kar sakde aan y_test naal.

jinna vi predict kitta hai usda mean kadea hai...

pred[name] = model.predict(X_test)
# Create a series with 1 being correct direction, 0 being wrong
# and then calculate the hit rate based on the actual direction
pred["%s_Correct" % name] = (1.0+pred[name]*pred["Actual"])/2.0
hit_rate = np.mean(pred["%s_Correct" % name])
print "%s: %.3f" % (name, hit_rate)

Eh dekho ja hor jyada deeply analyse karo ke oh Quantopian wali strategy ch kiddan Linear-regression use karke Stock-execution kitti hoi aa.. us ton pehlan oh sara code logistic regression wala apne hath naal try karo..



Hun eh dekhna hai vi eh linear ch tan moving average_factor consider karke decisoins bane

ne,, Fer logistics ch eh cheez kiven kamm aaogi. Us lai apan nu eh dekhna paina vi logistic-regression kis base te decision laindi hai.. ??

Logistics regression es base te decision laindi hai vi ajj price upper jauga ja hethan....

us 1 and -1 de concept nu Dhyaan naal samjho,, Bahut Dhyaan naal!!

ok eh concept bahut easy aa samjhan vaaste, Dekho jra dhyaan naal.

    we want to predict the direction of the closing price at day N based solely on price information known at day N−1. 

    2. An upward directional move means that the closing price at N is higher than the price at N−1, while a downward move implies a closing price at N lower than at N−1.

matlab apan kisse ik din da closing price find out karna hai, Okay,,, je apan nu "1" answer milda hai tan esda matlab hai ke es din da closing price kall de closing price nalon jyada hovega,, je apan nu -1 milda hai tan esda matlab ke es din da closing price kall de closing price nalon ghatt hovega. :D es base te apan jehde decisoin chahe oh lai sakde aan. :D :)

If we can determine the direction of movement in a manner that significantly exceeds a 50% hit rate, with low error and a good statistical significance, then we are on the road to forming a basic systematic trading strategy based on our forecasts.


If we can determine the direction of movement in a manner that significantly exceeds a 50% hit rate,---- apan training set lai data ditta jo ke X-train and Y-train hai.. fer apan model nu X-test data ditta and model ne data predict kitta, model duara "predict kitta hoya data" apan "Y-test" naal compare kitta othon apan eh determine karange vi kinna ku hit rate hai, nahi tan eh process dubara and dubara and dubara karange... (Ehi asli machine learning hai :) lots and lots of iteration until you get the success rate score you actually need)

In this implementation we have chosen to assign each day as "Up" if the probability exceeds 0.5. We could make use of a different threshold, but for simplicity I have chosen 0.5.

lag di value 5 lai hai, esnu upper thalle karke vi apan kamm chla sake aan ehi bass karna hai...

Tainu Quantopian da research platform use karna chahida hai.. and dekhna chahida hai ke research platform 
kiven use karna hai!!

'''
    Linear Regression Curves vs. Bollinger Bands
    If Close price is greater than average+n*deviation, go short
    If Close price is less than average+n*deviation, go long
    Both should close when you cross the average/mean
'''
import numpy as np
from scipy import stats
from quantopian.pipeline import Pipeline
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume


def initialize(context):
    # Create, register and name a pipeline in initialize.
    pipe = Pipeline()
    attach_pipeline(pipe, 'dollar_volume_10m_pipeline')

    # Construct a 100-day average dollar volume factor and add it to the pipeline.
    dollar_volume = AverageDollarVolume(window_length=100)
    pipe.add(dollar_volume, 'dollar_volume')

    #Create high dollar-volume filter to be the top 2% of stocks by dollar volume.
    high_dollar_volume = dollar_volume.percentile_between(99, 100)
    # Set the screen on the pipelines to filter out securities.
    pipe.set_screen(high_dollar_volume)

    context.dev_multiplier = 2
    context.max_notional = 1000000
    context.min_notional = -1000000
    context.days_traded = 0
    schedule_function(func=process_data_and_order, date_rule=date_rules.every_day())

def before_trading_start(context, data):
    # Pipeline_output returns the constructed dataframe.
    output = pipeline_output('dollar_volume_10m_pipeline')

    # sort the output. Most liquid stocks are at the top of the list,
    # and least liquid stocks are at the bottom
    sorted_output = output.sort('dollar_volume', ascending = False)

    context.my_securities = sorted_output.index

def process_data_and_order(context, data):
    context.days_traded += 1
    dev_mult = context.dev_multiplier
    notional = context.portfolio.positions_value
    # Calls get_linear so that moving_average has something to reference by the time it is called
    
    
    # here using results of linear regression, But how?? Bass pta lagg e jana ,, hune hune :D :D :)))))))))))))))))))
    
    # ethe liner use hoya haiga vaaaaaaaa..
    
    linear = get_linear(context, data)
    
    # Only checks every 20 days
    if context.days_traded%20 == 0:
        try:
            for stock in context.my_securities:
                close = data.current(stock, "close")
                
                # liner list cho stocks nu fetch kitta hoya only,,,, hun usnu as a moving_average consider kitta hai...
                
                # What is moving average---Ok Liner regression nu use karke Apan Moving average kadho hai..!!
                
                moving_average = linear[stock]
                stddev_history = data.history(stock, "price", 20, "1d")[:-1]
                moving_dev = stddev_history.std()
                
                
                
                # Hun tu pehlan eh dekhna hai vi Linear and moving average vich ki sambandh hai.. (Hacker's way)
                
              
              
                # ethe bolinger Band Create kitta hoya hai,,, 
                
                band = moving_average + dev_mult*moving_dev
              
              # es band base te e bass saare decisoin laye gye hannnn.....!!
              
              
              
                # If close price is greater than band, short 5000 and if less, buy 5000
                if close > band and notional > context.min_notional:
                    order(stock, -5000)
                    log.debug("Shorting 5000 of " + str(stock))
                elif close < band and notional < context.max_notional:
                    order(stock, 5000)
                    log.debug("Going long 5000 of " + str(stock))
        except:
            return
# Linear regression curve that returns the intercept the curve
# Uses the past 20 days

def get_linear(context, data):
    days = [i for i in range(1,21)]
    # why created Dictionary of stocks?
    stocks = {}
    for stock in context.my_securities:
		
		# how linregress works, ay example of Linear-Regress? 
		# paste it here--------
		
        linear = stats.linregress(days, data.history(stock, "price", 20, "1d"))[1]
        stocks[stock] = linear
    return stocks
# sabh to pehlan ikk simple thing varify karo vi get_linear() function kiven kamm karda hai, 
