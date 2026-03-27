# GGTH-Predictions-2026
A machine learning system that predicts forex price movements using an ensemble of AI models (LSTM, Transformer, Light GBM, TCN, and GRU AI models)and provides a Metatrader 5 Expert Advisor the data it needs to make trades. It can be trained on any forex pair but testing shows it performs best on the EURUSD 15 minute chart with settings it already has by default. If anyone locates better settings please email them to me.

To install simply run the setup file and follow the instructions.
After setup just double click run_ggth_gui.bat file.

<strong>ATTENTION - In settings for the expert advisor TURN STRATEGY TESTER MODE TO FALSE TO SEE THE PREDICTIONS DISPLAYED ON YOUR CHART!!!! </strong>

IMPORTANT - for the system to work properly you need to be logged into the community portal on your Metatrader 5 platform: Click TOOLS - OPTIONS - COMMUNITY - then login.

the AI models can be trained on any Forex pair you desire. I would suggest you backtest any new forex pairs and settings before using. To backtest you need to run the Generate backtest to create a file that will contain what prices AI would have predicted in that timeframe. Then it can use those predictions during the strategy tester run.

<strong>IMPORTANT!!!!!!!!!!!!!!!!!!!!!</strong>
When backtesting, to keep the AI from cheating you MUST train it on different dates then your prediction file for backtesting for example train the AI on dates 2019-01-01 to 2023-12-31 then create a backtesting prediction file (this can be done in the GUI) for 2024-01-01 to 2026-2-22
Then run your strategy tester for the timeframe your backtest prediction files were created on. Also be sure to change it to "Backtesting Mode = True"  in the settings when using strategy tester
All of this can be done easily through the User interface.

Also, included is an expert advisor that adheres to FIFO regulations for United States traders to use.

Everything is easy to do and can be done through the GUI.

I'd also appreciate any feedback from anyone who tests or uses this. It would be very appredciated to get input for the upgraded system I'm going to be working on.

Good Luck!!
