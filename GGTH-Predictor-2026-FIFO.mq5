//+------------------------------------------------------------------+
//|                                                GGTH-Predictor.mq5  |
//|                                      Copyright 2026, Jason Rusk  |
//|                                       jason.w.rusk@gmail.com     |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Jason Rusk"
#property link      "jason.w.rusk@gmail.com"
#property version   "1.08"
#property description "GGTH ML Predictor EA - FIFO Compliant Edition"
#property description "ML predictions (1H/4H/1D) with MA100 trend filter."
#property description "10h max hold, 0-17h session, 15-pip averaging at 0.4 lots."
#property description "v1.07: FIFO compliant - oldest-first close, hedge prevention."
#property description "v1.08: Margin utilization guard - blocks new/avg positions above threshold."


#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Enumeration for lot sizing mode                                  |
//+------------------------------------------------------------------+
enum ENUM_LOT_MODE
  {
   LOT_MODE_FIXED=0,      // Fixed Lot Size
   LOT_MODE_RISK=1        // Risk Percent
  };

//+------------------------------------------------------------------+
//| Input parameters                                                  |
//+------------------------------------------------------------------+
//--- Testing Mode
input group "=== Testing Mode ==="
input bool    InpStrategyTesterMode=false;                  // Strategy Tester Mode (use CSV lookups)

//--- ML Prediction Settings
input group "=== ML Prediction Settings ==="
input string  InpSymbol="EURUSD";                           // Symbol to track
input ENUM_TIMEFRAMES InpTradingTimeframe=PERIOD_H1;        // Which prediction to use for trading
input bool    InpEnableTrading=true;                        // Enable live trading
input int     InpMinPredictionPips=14;                      // Min prediction distance to confirm trade

//--- Position Sizing
input group "=== Position Sizing ==="
input int     InpMagicNumber=111111;                          // Magic Number (unique per EA instance)
input ENUM_LOT_MODE InpLotMode=LOT_MODE_FIXED;              // Lot Size Mode
input double  InpFixedLotSize=0.01;                          // Fixed Lot Size (if using Fixed mode)
input double  InpRiskPercent=1.0;                           // Risk per trade % (if using Risk mode)
input bool    InpFIFOCompliant=true;                       // FIFO Compliant Mode (NFA/US brokers)

//--- Averaging Down Settings
input group "=== Averaging Down Settings ==="
input bool    InpUseAveragingDown=true;                     // Enable Averaging Down
input double  InpAvgLevel1Lots=0.02;                         // Level 1: Lot Size
input int     InpAvgLevel1Pips=35;                        // Level 1: Pips Against Position 
input double  InpAvgLevel2Lots=0.02;                         // Level 2: Lot Size
input int     InpAvgLevel2Pips=15;                          // Level 2: Pips Against Position
input double  InpAvgLevel3Lots=0.3;                         // Level 3: Lot Size
input int     InpAvgLevel3Pips=10000;                       // Level 3: Pips Against Position (10000=disabled)

//--- Margin Risk Settings
input group "=== Margin Risk Settings ==="
input double  InpMaxMarginUsagePct=80.0;                    // Max Margin Usage % (blocks new positions above this)

//--- Profit Protection Settings
input group "=== Profit Protection Settings ==="
input bool    InpUseProfitProtection=true;                  // Enable Profit Protection
input int     InpMinPositionsForProtection=2;               // Min Positions to Trigger Protection
input double  InpProfitTargetAmount=2;                     // Profit Target ($) to Close All

//--- Max Hold Time Settings
input group "=== Max Hold Time Settings ==="
input bool    InpUseMaxHoldTime=true;                       // Enable Max Hold Time (replaces stop loss)
input int     InpMaxHoldHours=10;                           // Max Hours to Hold Positions

//--- Market Context Veto Settings 
input group "=== Market Context Veto Settings ==="
input bool    InpUseMarketContextVeto=false;                // Use Market Context Veto System
input double  InpVolatilitySpikeMultiplier=2.5;             // ATR Spike Threshold (2.5x = extreme)
input double  InpMaxCandleChangePercent=0.5;                // Max Single Candle % Move
input int     InpVolatilityLookback=20;                     // Volatility Average Period

//--- Take Profit & Stop Loss
input group "=== Take Profit & Stop Loss ==="
input bool    InpUsePredictedPrice=true;                    // Use predicted price as TP
input int     InpStopLossPips=200;                          // Stop loss in pips
input int     InpTakeProfitPips=200;                        // Take profit in pips (if not using predicted)
input double  InpTPMultiplier=1.0;                          // TP multiplier (adjust predicted TP)
input int     InpMinTPPips=2;                               // Minimum TP distance in pips
input int     InpMaxTPPips=500;                             // Maximum TP distance in pips

//--- Trend Filter
input group "=== Trend Filter ==="
input bool    InpUseTrendFilter=true;                       // Use trend filter
input int     InpTrendMAPeriod=100;                         // Trend MA period
input ENUM_MA_METHOD InpTrendMAMethod=MODE_EMA;             // Trend MA method
input ENUM_APPLIED_PRICE InpTrendMAPrice=PRICE_CLOSE;       // Trend MA price

//--- RSI Filter
input group "=== RSI Filter ==="
input bool    InpUseRSIFilter=true;                         // Use RSI filter
input int     InpRSIPeriod=14;                              // RSI period
input double  InpRSIOverbought=70.0;                        // RSI overbought level
input double  InpRSIOversold=30.0;                          // RSI oversold level

//--- Trailing Stop
input group "=== Trailing Stop ==="
input bool    InpUseTrailingStop=false;                     // Enable trailing stop
input int     InpTrailingStopPips=12;                       // Trailing stop distance in pips
input int     InpTrailingStepPips=5;                        // Minimum price movement to trail (pips)

//--- Trading Days
input group "=== Trading Days ==="
input bool    InpTradeMonday=true;                          // Trade on Monday
input bool    InpTradeTuesday=true;                         // Trade on Tuesday
input bool    InpTradeWednesday=true;                       // Trade on Wednesday
input bool    InpTradeThursday=true;                        // Trade on Thursday
input bool    InpTradeFriday=true;                          // Trade on Friday
input bool    InpTradeSaturday=false;                       // Trade on Saturday
input bool    InpTradeSunday=false;                         // Trade on Sunday

//--- Trading Sessions
input group "=== Trading Sessions ==="
input bool    InpUseSession1=true;                          // Enable Session 1
input int     InpSession1StartHour=0;                       // Session 1 Start Hour (0-23)
input int     InpSession1StartMinute=0;                     // Session 1 Start Minute (0-59)
input int     InpSession1EndHour=17;                        // Session 1 End Hour (0-23)
input int     InpSession1EndMinute=0;                       // Session 1 End Minute (0-59)

input bool    InpUseSession2=false;                         // Enable Session 2
input int     InpSession2StartHour=8;                       // Session 2 Start Hour (0-23)
input int     InpSession2StartMinute=0;                     // Session 2 Start Minute (0-59)
input int     InpSession2EndHour=16;                        // Session 2 End Hour (0-23)
input int     InpSession2EndMinute=0;                       // Session 2 End Minute (0-59)

input bool    InpUseSession3=false;                         // Enable Session 3
input int     InpSession3StartHour=16;                      // Session 3 Start Hour (0-23)
input int     InpSession3StartMinute=0;                     // Session 3 Start Minute (0-59)
input int     InpSession3EndHour=23;                        // Session 3 End Hour (0-23)
input int     InpSession3EndMinute=59;                      // Session 3 End Minute (0-59)

//--- Display Settings (MAXIMUM SPACING)
input group "=== Display Settings ==="
input int     InpFontSize=14;                               // Font size (LARGE for easy reading)
input color   InpTextColor=clrWhite;                        // Text color
input color   InpUpColor=clrLimeGreen;                      // Up prediction color
input color   InpDownColor=clrRed;                          // Down prediction color
input int     InpXOffset=20;                                // X offset from left
input int     InpYOffset=30;                                // Y offset from top
input bool    InpShowDebug=true;                            // Show debug info

//--- Adaptive Learning Settings
input group "=== Adaptive Learning ==="
input bool    InpEnableAdaptiveLearning=false;              // Enable online adaptive learning
input int     InpAdaptLookback=20;                          // Rolling window: trades to evaluate
input int     InpAdaptEveryN=5;                             // Trigger adaptation every N closed trades
input double  InpAdaptRate=0.15;                            // Learning rate (0.05=slow, 0.30=fast)
input double  InpAdaptMinPredFloor=1.0;                     // Minimum allowed adaptive pred pips
input double  InpAdaptMinPredCeil=60.0;                     // Maximum allowed adaptive pred pips
input double  InpAdaptLotMultFloor=0.25;                    // Minimum lot multiplier
input double  InpAdaptLotMultCeil=1.5;                      // Maximum lot multiplier (capped at 1.5x)
input bool    InpAdaptResetOnInit=false;                    // Reset all learned params on init
input bool    InpShowAdaptiveDebug=true;                    // Show adaptive learning output

//+------------------------------------------------------------------+
//| Market Context Structure (FIXED)                                 |
//+------------------------------------------------------------------+
struct CMarketContext
  {
   bool              veto_active;
   string            reasons[];
   double            volatility_ratio;
   double            max_candle_change;
   datetime          last_check;
  };

//+------------------------------------------------------------------+
//| CSV Prediction Structure                                          |
//+------------------------------------------------------------------+
struct CCSVPrediction
  {
   datetime          timestamp;
   double            prediction;
   double            change_pct;
   double            ensemble_std;
  };

//+------------------------------------------------------------------+
//| Prediction Data Structure                                         |
//+------------------------------------------------------------------+
struct CPredictionData
  {
   double            prediction;
   double            change_pct;
   double            ensemble_std;
   datetime          last_update;
   bool              trade_allowed;
  };

//+------------------------------------------------------------------+
//| Prediction Record Structure                                       |
//+------------------------------------------------------------------+
struct CPredictionRecord
  {
   datetime          timestamp;
   double            predicted_price;
   double            start_price;
   bool              checked;
   bool              accurate;
   datetime          check_time;
   string            timeframe_name;
  };

//+------------------------------------------------------------------+
//| Accuracy Tracker Structure                                        |
//+------------------------------------------------------------------+
struct CAccuracyTracker
  {
   int               total_predictions;
   int               accurate_predictions;
   double            accuracy_percent;
   CPredictionRecord current_prediction;
  };

//+------------------------------------------------------------------+
//| Averaging State Structure                                         |
//+------------------------------------------------------------------+
struct CAveragingState
  {
   double            original_entry_price;
   double            original_take_profit;
   long              original_position_type;
   datetime          series_start_time;
   bool              level1_triggered;
   bool              level2_triggered;
   bool              level3_triggered;
  };

//+------------------------------------------------------------------+
//| Single closed trade record for adaptive learning                 |
//+------------------------------------------------------------------+
#define MAX_TRADE_HISTORY 200
#define MAX_OPEN_TRACKING  50

struct CTradeRecord
  {
   ulong             position_id;       // MT5 position ID (DEAL_POSITION_ID)
   datetime          open_time;
   datetime          close_time;
   double            profit;
   double            pred_change_pct;   // signal strength at entry
   double            pred_pips;         // predicted distance at entry
   double            rsi_at_entry;
   bool              was_buy;
   bool              won;               // profit > 0
   bool              used;              // slot occupied
  };

//--- Lightweight record kept while trade is still open
struct COpenTradeEntry
  {
   ulong             position_id;
   double            pred_change_pct;
   double            pred_pips;
   double            rsi_at_entry;
   bool              was_buy;
   bool              used;
  };

//+------------------------------------------------------------------+
//| Adaptive learning state (persisted to file)                      |
//+------------------------------------------------------------------+
struct CAdaptiveState
  {
   //--- Adapted parameters (these replace the equivalent Inp* values)
   double            min_pred_pips;     // adaptive InpMinPredictionPips
   double            lot_multiplier;    // multiplied onto base lot size
   double            rsi_overbought;    // adaptive RSI OB level
   double            rsi_oversold;      // adaptive RSI OS level

   //--- Rolling performance metrics (computed each adaptation)
   double            win_rate;
   double            profit_factor;
   double            avg_win;
   double            avg_loss;
   double            kelly_fraction;

   //--- Counters
   int               trades_since_last_adapt;
   int               total_adaptations;
   int               consecutive_losses;

   //--- Rolling trade history (circular buffer)
   CTradeRecord      history[MAX_TRADE_HISTORY];
   int               history_head;     // next write index (circular)
   int               history_size;     // how many valid entries (up to MAX)
  };

//+------------------------------------------------------------------+
//| GGTH Expert Advisor Class                                        |
//+------------------------------------------------------------------+
class CGGTHExpert
  {
private:
   //--- Trade management
   CTrade            m_trade;
   
   //--- Symbol and file management
   string            m_symbol;
   string            m_predictions_file;
   string            m_status_file;
   
   //--- Indicator handles
   int               m_handle_trend_ma;
   int               m_handle_rsi;
   
   //--- Prediction data
   CPredictionData   m_pred_1H;
   CPredictionData   m_pred_4H;
   CPredictionData   m_pred_1D;
   double            m_current_price;
   
   //--- Accuracy tracking
   CAccuracyTracker  m_tracker_1H;
   CAccuracyTracker  m_tracker_4H;
   CAccuracyTracker  m_tracker_1D;
   
   //--- Market context (FIXED)
   CMarketContext    m_market_context;
   
   //--- Averaging state
   CAveragingState   m_avg_state;
   
   //--- CSV data for backtesting
   CCSVPrediction    m_csv_1H[];
   CCSVPrediction    m_csv_4H[];
   CCSVPrediction    m_csv_1D[];
   int               m_csv_1H_count;
   int               m_csv_4H_count;
   int               m_csv_1D_count;
   
   //--- State variables
   datetime          m_last_pred_bar_time;   // Last bar time on prediction timeframe
   datetime          m_last_chart_bar_time;  // Last bar time on chart timeframe (Period())
   datetime          m_last_trade_time;
   int               m_min_trade_interval;

   //--- Adaptive learning state
   CAdaptiveState    m_adaptive;
   COpenTradeEntry   m_open_tracking[MAX_OPEN_TRACKING]; // pending open trades

public:
   //--- Constructor/Destructor
                     CGGTHExpert();
                    ~CGGTHExpert();
   
   //--- Main interface methods
   int               Init();
   void              Deinit();
   void              OnTick();
   void              OnTradeTransaction(const MqlTradeTransaction &trans,
                                        const MqlTradeRequest &request,
                                        const MqlTradeResult &result);

private:
   //--- Initialization methods
   void              InitializeTrackers();
   bool              InitializeIndicators();
   bool              LoadCSVBacktestData();
   
   //--- Event handlers
   void              OnNewBar();
   void              OnNewChartBar();
   
   //--- Prediction loading
   bool              LoadPredictionsFromJSON();
   bool              LoadPredictionsFromCSV();
   bool              ParsePredictionJSON(string json,string timeframe,CPredictionData &pred);
   bool              LoadCSVLookupFile(ENUM_TIMEFRAMES timeframe);
   
   //--- Trading logic
   void              CheckForTradeSignal();
   bool              IsTradingAllowed();
   bool              IsWithinTradingSession(int hour,int minute);
   
   //--- Filters
   bool              CheckTrendFilter(bool &signal_buy,bool &signal_sell);
   bool              CheckRSIFilter(bool &signal_buy,bool &signal_sell);
   
   //--- Market context (FIXED - volatility-based)
   void              UpdateMarketContext();
   
   //--- Position management
   int               CountOpenPositions();
   double            GetTotalProfit();
   bool              GetFirstPositionInfo(double &entry_price,long &pos_type,datetime &open_time,double &take_profit);
   bool              CloseAllPositions(string reason);
   double            CalculateLotSize();
   
   //--- Averaging down
   void              CheckAveragingDown();
   bool              ExecuteAveragingOrder(int level,double lots);
   void              ResetAveragingState();
   
   //--- Profit protection and management
   void              CheckProfitProtection();
   void              CheckMaxHoldTime();
   void              ApplyTrailingStop();
   
   //--- Accuracy tracking
   void              UpdateAccuracyTracking();
   void              CheckAccuracyForTimeframe(CAccuracyTracker &tracker,CPredictionData &pred,ENUM_TIMEFRAMES tf);
   void              SaveAccuracyData();
   void              LoadAccuracyData();
   
   //--- Display methods (MAXIMUM SPACING)
   void              DisplayInfo();
   void              DisplayPredictionLine(string tf_name,CPredictionData &pred,CAccuracyTracker &tracker,int x_pos,int &y_pos);
   void              DisplayError();
   void              CreateLabel(string name,int x,int y,string text,int font_size,color clr);

   //--- Adaptive learning
   void              InitAdaptiveState();
   void              RecordTradeEntry(ulong position_id,bool is_buy,double pred_change_pct,double pred_pips,double rsi);
   void              ProcessClosedTrade(ulong position_id,double profit);
   void              AdaptParameters();
   void              ComputeRollingMetrics(double &win_rate,double &profit_factor,
                                           double &avg_win,double &avg_loss);
   double            ComputeKellyFraction(double win_rate,double avg_win,double avg_loss);
   void              SaveAdaptiveState();
   void              LoadAdaptiveState();
   void              DisplayAdaptiveInfo(int x_pos,int &y_pos,int line_height);
  };

//--- Global instance of expert
CGGTHExpert g_expert;

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
  {
   return(g_expert.Init());
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   g_expert.Deinit();
  }

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
  {
   g_expert.OnTick();
  }

//+------------------------------------------------------------------+
//| Trade transaction handler - routes to adaptive learner           |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
  {
   g_expert.OnTradeTransaction(trans,request,result);
  }

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CGGTHExpert::CGGTHExpert() : m_symbol(InpSymbol),
                             m_handle_trend_ma(INVALID_HANDLE),
                             m_handle_rsi(INVALID_HANDLE),
                             m_current_price(0),
                             m_csv_1H_count(0),
                             m_csv_4H_count(0),
                             m_csv_1D_count(0),
                             m_last_pred_bar_time(0),
                             m_last_chart_bar_time(0),
                             m_last_trade_time(0),
                             m_min_trade_interval(60)
  {
  }

//+------------------------------------------------------------------+
//| Destructor                                                        |
//+------------------------------------------------------------------+
CGGTHExpert::~CGGTHExpert()
  {
  }

//+------------------------------------------------------------------+
//| Initialization and checking for input parameters                 |
//+------------------------------------------------------------------+
int CGGTHExpert::Init()
  {
//--- Set up file paths
   m_predictions_file=m_symbol+"_predictions_multitf.json";
   m_status_file=m_symbol+"_status.json";

//--- Initialize tracking structures
   InitializeTrackers();

//--- Load saved accuracy data
   LoadAccuracyData();

//--- Create indicator handles
   if(!InitializeIndicators())
     {
      Print("Error: Failed to initialize indicators");
      return(INIT_FAILED);
     }

//--- Load CSV backtest data if in tester mode
   if(InpStrategyTesterMode)
     {
      if(!LoadCSVBacktestData())
        {
         Print("Error: Failed to load CSV backtest data");
         return(INIT_FAILED);
        }
     }

//--- Initialize averaging state
   ResetAveragingState();

//--- Initialize adaptive learning state
   InitAdaptiveState();
   if(!InpAdaptResetOnInit)
      LoadAdaptiveState();

//--- Initialize market context (FIXED)
   m_market_context.veto_active=false;
   m_market_context.volatility_ratio=0;
   m_market_context.max_candle_change=0;
   m_market_context.last_check=0;
   ArrayResize(m_market_context.reasons,0);

//--- Set magic number
   m_trade.SetExpertMagicNumber(InpMagicNumber);

   Print("GGTH Predictor EA v1.07 (FIFO Compliant) initialized | Magic: ",InpMagicNumber,
         " | FIFO mode: ",(InpFIFOCompliant?"ON":"OFF"));
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Deinitialization                                                  |
//+------------------------------------------------------------------+
void CGGTHExpert::Deinit()
  {
//--- Save accuracy data
   SaveAccuracyData();

//--- Save adaptive learning state
   if(InpEnableAdaptiveLearning)
      SaveAdaptiveState();

//--- Release indicator handles
   if(m_handle_trend_ma!=INVALID_HANDLE)
      IndicatorRelease(m_handle_trend_ma);
   if(m_handle_rsi!=INVALID_HANDLE)
      IndicatorRelease(m_handle_rsi);

//--- Remove all chart objects
   ObjectsDeleteAll(0,"MLEA_");

   Print("GGTH Predictor EA v1.07 deinitialized");
  }

//+------------------------------------------------------------------+
//| Main tick handler                                                 |
//+------------------------------------------------------------------+
void CGGTHExpert::OnTick()
  {
//--- Update current price
   m_current_price=SymbolInfoDouble(m_symbol,SYMBOL_BID);

//--- Check for new PREDICTION TIMEFRAME bar (update signal / accuracy tracking)
   datetime pred_bar_time=iTime(m_symbol,InpTradingTimeframe,0);
   if(pred_bar_time!=m_last_pred_bar_time)
     {
      m_last_pred_bar_time=pred_bar_time;
      OnNewBar();
     }

//--- Check for new CHART TIMEFRAME bar (fire a trade on every bar while signal is active)
   datetime chart_bar_time=iTime(m_symbol,Period(),0);
   if(chart_bar_time!=m_last_chart_bar_time)
     {
      m_last_chart_bar_time=chart_bar_time;
      OnNewChartBar();
     }

//--- Load predictions
   bool predictions_loaded=false;
   if(InpStrategyTesterMode)
      predictions_loaded=LoadPredictionsFromCSV();
   else
      predictions_loaded=LoadPredictionsFromJSON();

//--- Update display
   if(predictions_loaded)
      DisplayInfo();
   else
      DisplayError();

//--- Check profit protection
   if(InpUseProfitProtection)
      CheckProfitProtection();

//--- Check max hold time
   if(InpUseMaxHoldTime)
      CheckMaxHoldTime();

//--- Apply trailing stop if enabled
   if(InpUseTrailingStop)
      ApplyTrailingStop();

//--- Check for averaging down opportunities
   if(InpUseAveragingDown)
      CheckAveragingDown();
  }


//+------------------------------------------------------------------+
//| Prediction timeframe bar handler - update signal & tracking      |
//+------------------------------------------------------------------+
void CGGTHExpert::OnNewBar()
  {
//--- Update market context if veto system enabled (FIXED VERSION)
   if(InpUseMarketContextVeto)
      UpdateMarketContext();

//--- Update accuracy tracking
   UpdateAccuracyTracking();

//--- BELT-AND-SUSPENDERS: enforce max hold time on every new bar too.
//--- This catches edge cases where session boundaries may have delayed
//--- the tick-level timer check (e.g. post-weekend gap opens).
   if(InpUseMaxHoldTime)
      CheckMaxHoldTime();
  }

//+------------------------------------------------------------------+
//| Chart timeframe bar handler - fire trade on every bar            |
//+------------------------------------------------------------------+
void CGGTHExpert::OnNewChartBar()
  {
//--- Place a trade on this chart bar if signal conditions are met
   if(InpEnableTrading)
      CheckForTradeSignal();
  }

//+------------------------------------------------------------------+
//| Initialize tracking structures                                    |
//+------------------------------------------------------------------+
void CGGTHExpert::InitializeTrackers()
  {
//--- Initialize H1 tracker
   m_tracker_1H.total_predictions=0;
   m_tracker_1H.accurate_predictions=0;
   m_tracker_1H.accuracy_percent=0.0;
   m_tracker_1H.current_prediction.checked=false;
   m_tracker_1H.current_prediction.timeframe_name="H1";

//--- Initialize H4 tracker
   m_tracker_4H.total_predictions=0;
   m_tracker_4H.accurate_predictions=0;
   m_tracker_4H.accuracy_percent=0.0;
   m_tracker_4H.current_prediction.checked=false;
   m_tracker_4H.current_prediction.timeframe_name="H4";

//--- Initialize D1 tracker
   m_tracker_1D.total_predictions=0;
   m_tracker_1D.accurate_predictions=0;
   m_tracker_1D.accuracy_percent=0.0;
   m_tracker_1D.current_prediction.checked=false;
   m_tracker_1D.current_prediction.timeframe_name="D1";
  }

//+------------------------------------------------------------------+
//| Initialize indicators                                             |
//+------------------------------------------------------------------+
bool CGGTHExpert::InitializeIndicators()
  {
//--- Create trend MA handle
   if(InpUseTrendFilter)
     {
      m_handle_trend_ma=iMA(m_symbol,InpTradingTimeframe,InpTrendMAPeriod,
                            0,InpTrendMAMethod,InpTrendMAPrice);
      if(m_handle_trend_ma==INVALID_HANDLE)
        {
         Print("Error creating Trend MA indicator");
         return(false);
        }
     }

//--- Create RSI handle
   if(InpUseRSIFilter)
     {
      m_handle_rsi=iRSI(m_symbol,InpTradingTimeframe,InpRSIPeriod,PRICE_CLOSE);
      if(m_handle_rsi==INVALID_HANDLE)
        {
         Print("Error creating RSI indicator");
         return(false);
        }
     }

   return(true);
  }

//+------------------------------------------------------------------+
//| Reset averaging state structure                                   |
//+------------------------------------------------------------------+
void CGGTHExpert::ResetAveragingState()
  {
   m_avg_state.original_entry_price=0;
   m_avg_state.original_take_profit=0;
   m_avg_state.original_position_type=-1;
   m_avg_state.series_start_time=0;
   m_avg_state.level1_triggered=false;
   m_avg_state.level2_triggered=false;
   m_avg_state.level3_triggered=false;
  }

//+------------------------------------------------------------------+
//| Execute averaging order                                           |
//+------------------------------------------------------------------+
bool CGGTHExpert::ExecuteAveragingOrder(int level,double lots)
  {
//--- FIFO COMPLIANCE: averaging must always match the existing position direction.
//--- This is already guaranteed by using m_avg_state.original_position_type,
//--- but if FIFO mode is on we double-check before sending the order.
   if(InpFIFOCompliant)
     {
      for(int i=PositionsTotal()-1; i>=0; i--)
        {
         ulong chk=PositionGetTicket(i);
         if(chk<=0) continue;
         if(PositionGetString(POSITION_SYMBOL)!=m_symbol) continue;
         if(PositionGetInteger(POSITION_MAGIC)!=InpMagicNumber) continue;
         long existing_type=PositionGetInteger(POSITION_TYPE);
         if(existing_type!=m_avg_state.original_position_type)
           {
            Print("[FIFO] Averaging order blocked - direction mismatch detected");
            return(false);
           }
        }
     }

//--- Calculate lot size
   double lot_size=CalculateLotSize();
   if(lot_size<=0)
      lot_size=lots;

//--- Get current prices
   double ask=SymbolInfoDouble(m_symbol,SYMBOL_ASK);
   double bid=SymbolInfoDouble(m_symbol,SYMBOL_BID);

//--- Use original position's TP for averaging orders
   double tp=m_avg_state.original_take_profit;
   double sl=0;

   string comment=StringFormat("GGTH v1.06 [AVG L%d] TP:%."+IntegerToString(_Digits)+"f",level,tp);

//--- Execute order based on position type
   if(m_avg_state.original_position_type==POSITION_TYPE_BUY)
     {
      if(m_trade.Buy(lot_size,m_symbol,ask,sl,tp,comment))
        {
         Print("[AVG] Averaging DOWN - BUY Level ",level," at ",ask," TP: ",tp);
         return(true);
        }
     }
   else if(m_avg_state.original_position_type==POSITION_TYPE_SELL)
     {
      if(m_trade.Sell(lot_size,m_symbol,bid,sl,tp,comment))
        {
         Print("[AVG] Averaging DOWN - SELL Level ",level," at ",bid," TP: ",tp);
         return(true);
        }
     }

   return(false);
  }

//+------------------------------------------------------------------+
//| Calculate lot size based on mode                                  |
//+------------------------------------------------------------------+
double CGGTHExpert::CalculateLotSize()
  {
   double lot_size=0;

   if(InpLotMode==LOT_MODE_FIXED)
     {
      lot_size=InpFixedLotSize;
     }
   else if(InpLotMode==LOT_MODE_RISK)
     {
      double balance=AccountInfoDouble(ACCOUNT_BALANCE);
      double risk_amount=balance*(InpRiskPercent/100.0);
      double point=SymbolInfoDouble(m_symbol,SYMBOL_POINT);
      double pip=point;
      if(_Digits==3 || _Digits==5)
         pip=point*10.0;

      double sl_distance=InpStopLossPips*pip;
      double tick_value=SymbolInfoDouble(m_symbol,SYMBOL_TRADE_TICK_VALUE);

      if(sl_distance>0 && tick_value>0)
        {
         lot_size=(risk_amount/sl_distance)/tick_value;
        }
     }

//--- Normalize lot size
   double min_lot=SymbolInfoDouble(m_symbol,SYMBOL_VOLUME_MIN);
   double max_lot=SymbolInfoDouble(m_symbol,SYMBOL_VOLUME_MAX);
   double lot_step=SymbolInfoDouble(m_symbol,SYMBOL_VOLUME_STEP);

   lot_size=MathFloor(lot_size/lot_step)*lot_step;
   lot_size=MathMax(lot_size,min_lot);
   lot_size=MathMin(lot_size,max_lot);

   return(lot_size);
  }

//+------------------------------------------------------------------+
//| Count open positions for symbol                                   |
//+------------------------------------------------------------------+
int CGGTHExpert::CountOpenPositions()
  {
   int count=0;
   for(int i=PositionsTotal()-1; i>=0; i--)
     {
      ulong ticket=PositionGetTicket(i);
      if(ticket>0)
        {
         if(PositionGetString(POSITION_SYMBOL)==m_symbol &&
            PositionGetInteger(POSITION_MAGIC)==InpMagicNumber)
           {
            count++;
           }
        }
     }
   return count;
  }

//+------------------------------------------------------------------+
//| Get total profit of all positions for symbol                      |
//+------------------------------------------------------------------+
double CGGTHExpert::GetTotalProfit()
  {
   double total_profit=0;
   for(int i=PositionsTotal()-1; i>=0; i--)
     {
      ulong ticket=PositionGetTicket(i);
      if(ticket>0)
        {
         if(PositionGetString(POSITION_SYMBOL)==m_symbol &&
            PositionGetInteger(POSITION_MAGIC)==InpMagicNumber)
           {
            total_profit+=PositionGetDouble(POSITION_PROFIT);
            total_profit+=PositionGetDouble(POSITION_SWAP);
           }
        }
     }
   return total_profit;
  }

//+------------------------------------------------------------------+
//| Get first position info for averaging                             |
//+------------------------------------------------------------------+
bool CGGTHExpert::GetFirstPositionInfo(double &entry_price,long &pos_type,datetime &open_time,double &take_profit)
  {
   datetime earliest_time=D'2099.12.31';
   bool found=false;

   for(int i=PositionsTotal()-1; i>=0; i--)
     {
      ulong ticket=PositionGetTicket(i);
      if(ticket>0)
        {
         if(PositionGetString(POSITION_SYMBOL)==m_symbol &&
            PositionGetInteger(POSITION_MAGIC)==InpMagicNumber)
           {
            datetime pos_time=(datetime)PositionGetInteger(POSITION_TIME);
            if(pos_time<earliest_time)
              {
               earliest_time=pos_time;
               entry_price=PositionGetDouble(POSITION_PRICE_OPEN);
               pos_type=PositionGetInteger(POSITION_TYPE);
               open_time=pos_time;
               take_profit=PositionGetDouble(POSITION_TP);
               found=true;
              }
           }
        }
     }
   return found;
  }

//+------------------------------------------------------------------+
//| Close all positions for symbol                                    |
//+------------------------------------------------------------------+
bool CGGTHExpert::CloseAllPositions(string reason)
  {
   bool all_closed=true;
   int closed_count=0;
   double total_profit=GetTotalProfit();

   //--- FIFO COMPLIANCE: collect matching tickets and sort by open time (oldest first)
   //--- This satisfies the NFA FIFO rule which requires positions to be closed
   //--- in the same order they were opened (first-in, first-out).
   ulong    tickets[];
   datetime times[];
   int      match_count=0;

   ArrayResize(tickets,PositionsTotal());
   ArrayResize(times,  PositionsTotal());

   for(int i=0; i<PositionsTotal(); i++)
     {
      ulong ticket=PositionGetTicket(i);
      if(ticket>0)
        {
         if(PositionGetString(POSITION_SYMBOL)==m_symbol &&
            PositionGetInteger(POSITION_MAGIC)==InpMagicNumber)
           {
            tickets[match_count]=(ulong)ticket;
            times[match_count]  =(datetime)PositionGetInteger(POSITION_TIME);
            match_count++;
           }
        }
     }

   //--- Insertion-sort by open time ascending (oldest first)
   for(int a=1; a<match_count; a++)
     {
      ulong    t_key =tickets[a];
      datetime dt_key=times[a];
      int b=a-1;
      while(b>=0 && times[b]>dt_key)
        {
         tickets[b+1]=tickets[b];
         times[b+1]  =times[b];
         b--;
        }
      tickets[b+1]=t_key;
      times[b+1]  =dt_key;
     }

   //--- Close in FIFO order (oldest position first)
   for(int i=0; i<match_count; i++)
     {
      if(m_trade.PositionClose(tickets[i]))
        {
         closed_count++;
        }
      else
        {
         all_closed=false;
         Print("ERROR: Failed to close position ",tickets[i]," Error: ",GetLastError());
        }
     }

   if(closed_count>0)
     {
      Print("[OK] ",reason," - Closed ",closed_count," positions | Total P/L: $",
            DoubleToString(total_profit,2));
      ResetAveragingState();
     }

   return all_closed;
  }

//+------------------------------------------------------------------+
//| Check profit protection                                           |
//+------------------------------------------------------------------+
void CGGTHExpert::CheckProfitProtection()
  {
   if(!InpUseProfitProtection)
      return;

   int position_count=CountOpenPositions();

//--- Only trigger if we have minimum required positions
   if(position_count<InpMinPositionsForProtection)
      return;

   double total_profit=GetTotalProfit();

//--- Check if profit target reached
   if(total_profit>=InpProfitTargetAmount)
     {
      string reason=StringFormat("PROFIT PROTECTION: $%.2f profit with %d positions",
                                 total_profit,position_count);
      CloseAllPositions(reason);
     }
  }

//+------------------------------------------------------------------+
//| Check max hold time                                               |
//+------------------------------------------------------------------+
void CGGTHExpert::CheckMaxHoldTime()
  {
   if(!InpUseMaxHoldTime)
      return;

   int position_count=CountOpenPositions();
   if(position_count==0)
      return;

   //--- Use TimeCurrent() for wall-clock elapsed time.
   //--- This is UNCONDITIONAL - session boundaries do NOT pause the timer.
   datetime current_time=TimeCurrent();
   long max_seconds=(long)InpMaxHoldHours*3600;
   bool found_expired=false;

//--- Scan ALL our positions; close entire campaign if any single position is expired.
   for(int i=PositionsTotal()-1; i>=0; i--)
     {
      ulong ticket=PositionGetTicket(i);
      if(ticket<=0) continue;

      if(PositionGetString(POSITION_SYMBOL)!=m_symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC)!=InpMagicNumber) continue;

      datetime open_time=(datetime)PositionGetInteger(POSITION_TIME);

      //--- Guard against broker/backtest clock anomalies
      if(open_time<=0 || open_time>current_time) continue;

      long hold_seconds=(long)(current_time - open_time);

      if(hold_seconds >= max_seconds)
        {
         found_expired=true;
         Print(StringFormat("[TIMER] Position #%I64u held %d min (max %d h) - closing all",
                            ticket, (int)(hold_seconds/60), InpMaxHoldHours));
         break;
        }
     }

   if(found_expired)
     {
      string reason=StringFormat("MAX HOLD TIME: Position(s) exceeded %d hours",InpMaxHoldHours);
      CloseAllPositions(reason);
     }
  }

//+------------------------------------------------------------------+
//| Check averaging down                                              |
//+------------------------------------------------------------------+
void CGGTHExpert::CheckAveragingDown()
  {
   if(!InpUseAveragingDown)
      return;

//--- Check margin utilization - block averaging if margin usage exceeds threshold
   double avg_margin_used  = AccountInfoDouble(ACCOUNT_MARGIN);
   double avg_equity       = AccountInfoDouble(ACCOUNT_EQUITY);
   double avg_margin_usage = (avg_equity > 0) ? (avg_margin_used / avg_equity * 100.0) : 0.0;
   if(avg_margin_usage > InpMaxMarginUsagePct)
     {
      if(InpShowDebug)
         Print("[MARGIN] Averaging blocked: margin usage ",DoubleToString(avg_margin_usage,1),
               "% exceeds limit of ",DoubleToString(InpMaxMarginUsagePct,1),"%");
      return;
     }

   int position_count=CountOpenPositions();

//--- If no positions, reset state
   if(position_count==0)
     {
      if(m_avg_state.original_entry_price>0)
        {
         ResetAveragingState();
         if(InpShowDebug)
            Print("Averaging state reset - no positions");
        }
      return;
     }

//--- Get first position info (including take profit)
   double entry_price;
   long pos_type;
   datetime open_time;
   double take_profit;

   if(!GetFirstPositionInfo(entry_price,pos_type,open_time,take_profit))
      return;

//--- Initialize state if this is a new position series
   if(m_avg_state.series_start_time!=open_time)
     {
      ResetAveragingState();
      m_avg_state.original_entry_price=entry_price;
      m_avg_state.original_take_profit=take_profit;
      m_avg_state.original_position_type=pos_type;
      m_avg_state.series_start_time=open_time;

      if(InpShowDebug)
         Print("New position series started at ",entry_price," with TP at ",take_profit);
     }

//--- Calculate pip value
   double point=SymbolInfoDouble(m_symbol,SYMBOL_POINT);
   double pip=point;
   if(_Digits==3 || _Digits==5)
      pip=point*10.0;

   double current_bid=SymbolInfoDouble(m_symbol,SYMBOL_BID);
   double current_ask=SymbolInfoDouble(m_symbol,SYMBOL_ASK);

//--- Calculate how far price has moved against position
   double pips_against=0;

   if(m_avg_state.original_position_type==POSITION_TYPE_BUY)
     {
      pips_against=(m_avg_state.original_entry_price-current_bid)/pip;
     }
   else if(m_avg_state.original_position_type==POSITION_TYPE_SELL)
     {
      pips_against=(current_ask-m_avg_state.original_entry_price)/pip;
     }

//--- Only average down if price moved against us
   if(pips_against<=0)
      return;

//--- Check Level 1
   if(!m_avg_state.level1_triggered && pips_against>=InpAvgLevel1Pips)
     {
      if(ExecuteAveragingOrder(1,InpAvgLevel1Lots))
        {
         m_avg_state.level1_triggered=true;
         Print("[AVG] Level 1 triggered at ",DoubleToString(pips_against,1)," pips against");
        }
     }

//--- Check Level 2
   if(!m_avg_state.level2_triggered && pips_against>=InpAvgLevel2Pips)
     {
      if(ExecuteAveragingOrder(2,InpAvgLevel2Lots))
        {
         m_avg_state.level2_triggered=true;
         Print("[AVG] Level 2 triggered at ",DoubleToString(pips_against,1)," pips against");
        }
     }

//--- Check Level 3
   if(!m_avg_state.level3_triggered && pips_against>=InpAvgLevel3Pips)
     {
      if(ExecuteAveragingOrder(3,InpAvgLevel3Lots))
        {
         m_avg_state.level3_triggered=true;
         Print("[AVG] Level 3 triggered at ",DoubleToString(pips_against,1)," pips against");
        }
     }
  }

//+------------------------------------------------------------------+
//| Apply trailing stop                                               |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Apply trailing stop - PROFIT ONLY + REMOVE TP VERSION            |
//+------------------------------------------------------------------+
void CGGTHExpert::ApplyTrailingStop()
  {
   if(!InpUseTrailingStop)
      return;

   double point=SymbolInfoDouble(m_symbol,SYMBOL_POINT);
   double pip=point;
   if(_Digits==3 || _Digits==5)
      pip=point*10.0;

   double trailing_stop_distance=InpTrailingStopPips*pip;
   double trailing_step=InpTrailingStepPips*pip;

   for(int i=PositionsTotal()-1; i>=0; i--)
     {
      ulong ticket=PositionGetTicket(i);
      if(ticket>0)
        {
         if(PositionGetString(POSITION_SYMBOL)==m_symbol &&
            PositionGetInteger(POSITION_MAGIC)==InpMagicNumber)
           {
            long pos_type=PositionGetInteger(POSITION_TYPE);
            double pos_open=PositionGetDouble(POSITION_PRICE_OPEN);
            double pos_sl=PositionGetDouble(POSITION_SL);
            double pos_tp=PositionGetDouble(POSITION_TP);

            double bid=SymbolInfoDouble(m_symbol,SYMBOL_BID);
            double ask=SymbolInfoDouble(m_symbol,SYMBOL_ASK);

            if(pos_type==POSITION_TYPE_BUY)
              {
               //--- Calculate current profit in pips
               double profit_pips=(bid-pos_open)/pip;
               
               //--- ONLY trail if position is in profit
               if(profit_pips<=0)
                  continue;  // Skip this position, not in profit yet

               double new_sl=bid-trailing_stop_distance;
               
               //--- Check if we should move the stop
               if(new_sl>pos_sl && new_sl>pos_open && (new_sl-pos_sl)>=trailing_step)
                 {
                  //--- Remove TP when trailing engages (set to 0)
                  double new_tp=0;
                  
                  if(m_trade.PositionModify(ticket,new_sl,new_tp))
                    {
                     if(pos_tp>0 && InpShowDebug)
                       {
                        Print("[TRAIL] ENGAGED for BUY #",ticket);
                        Print("  Profit: +",DoubleToString(profit_pips,1)," pips");
                        Print("  New SL: ",new_sl," | TP REMOVED");
                       }
                     else if(InpShowDebug)
                       {
                        Print("[TRAIL] Updated for BUY #",ticket," to ",new_sl);
                       }
                    }
                  else
                    {
                     if(InpShowDebug)
                        Print("ERROR: Failed to modify BUY position #",ticket," Error: ",GetLastError());
                    }
                 }
              }
            else if(pos_type==POSITION_TYPE_SELL)
              {
               //--- Calculate current profit in pips
               double profit_pips=(pos_open-ask)/pip;
               
               //--- ONLY trail if position is in profit
               if(profit_pips<=0)
                  continue;  // Skip this position, not in profit yet

               double new_sl=ask+trailing_stop_distance;
               
               //--- Check if we should move the stop
               if((pos_sl==0 || new_sl<pos_sl) && new_sl<pos_open && (pos_sl-new_sl)>=trailing_step)
                 {
                  //--- Remove TP when trailing engages (set to 0)
                  double new_tp=0;
                  
                  if(m_trade.PositionModify(ticket,new_sl,new_tp))
                    {
                     if(pos_tp>0 && InpShowDebug)
                       {
                        Print("[TRAIL] ENGAGED for SELL #",ticket);
                        Print("  Profit: +",DoubleToString(profit_pips,1)," pips");
                        Print("  New SL: ",new_sl," | TP REMOVED");
                       }
                     else if(InpShowDebug)
                       {
                        Print("[TRAIL] Updated for SELL #",ticket," to ",new_sl);
                       }
                    }
                  else
                    {
                     if(InpShowDebug)
                        Print("ERROR: Failed to modify SELL position #",ticket," Error: ",GetLastError());
                    }
                 }
              }
           }
        }
     }
  }



//+------------------------------------------------------------------+
//| Update market context - FIXED VOLATILITY-BASED VERSION           |
//+------------------------------------------------------------------+
void CGGTHExpert::UpdateMarketContext()
  {
   if(!InpUseMarketContextVeto)
      return;

//--- Reset veto state
   m_market_context.veto_active=false;
   ArrayResize(m_market_context.reasons,0);
   m_market_context.volatility_ratio=0;
   m_market_context.max_candle_change=0;

//--- CHECK 1: Volatility Spike Detection (Risk-Off Events)
   double atr_current_buffer[];
   double atr_avg_buffer[];
   ArraySetAsSeries(atr_current_buffer,true);
   ArraySetAsSeries(atr_avg_buffer,true);

//--- Get current H1 ATR
   int atr_h1_handle=iATR(m_symbol,PERIOD_H1,14);
   if(atr_h1_handle==INVALID_HANDLE)
     {
      Print("ERROR: Cannot create ATR indicator for volatility check");
      return;
     }

   if(CopyBuffer(atr_h1_handle,0,0,1,atr_current_buffer)<1)
     {
      IndicatorRelease(atr_h1_handle);
      return;
     }

   double atr_current=atr_current_buffer[0];
   IndicatorRelease(atr_h1_handle);

//--- Get average ATR over longer period
   int atr_avg_handle=iATR(m_symbol,PERIOD_H4,InpVolatilityLookback);
   if(atr_avg_handle==INVALID_HANDLE)
     {
      Print("ERROR: Cannot create ATR indicator for average");
      return;
     }

   if(CopyBuffer(atr_avg_handle,0,0,1,atr_avg_buffer)<1)
     {
      IndicatorRelease(atr_avg_handle);
      return;
     }

   double atr_average=atr_avg_buffer[0];
   IndicatorRelease(atr_avg_handle);

//--- Calculate volatility ratio
   if(atr_average>0)
     {
      m_market_context.volatility_ratio=atr_current/atr_average;

      if(m_market_context.volatility_ratio>=InpVolatilitySpikeMultiplier)
        {
         m_market_context.veto_active=true;
         int size=ArraySize(m_market_context.reasons);
         ArrayResize(m_market_context.reasons,size+1);
         m_market_context.reasons[size]=StringFormat("Volatility Spike (%.1fx normal)",
                                                      m_market_context.volatility_ratio);
        }
     }

//--- CHECK 2: Rapid Price Movement Detection
   double close_prices[];
   ArraySetAsSeries(close_prices,true);

   int bars_copied=CopyClose(m_symbol,Period(),0,10,close_prices);

   if(bars_copied>=10)
     {
      double max_change_pct=0;

      for(int i=0; i<9; i++)
        {
         if(close_prices[i+1]>0)
           {
            double change_pct=MathAbs((close_prices[i]-close_prices[i+1])/close_prices[i+1])*100.0;
            if(change_pct>max_change_pct)
               max_change_pct=change_pct;
           }
        }

      m_market_context.max_candle_change=max_change_pct;

      if(max_change_pct>=InpMaxCandleChangePercent)
        {
         m_market_context.veto_active=true;
         int size=ArraySize(m_market_context.reasons);
         ArrayResize(m_market_context.reasons,size+1);
         m_market_context.reasons[size]=StringFormat("Rapid Price Movement (%.2f%% candle)",
                                                      max_change_pct);
        }
     }

//--- CHECK 3: Gap Detection
   if(bars_copied>=2)
     {
      double current_open=iOpen(m_symbol,Period(),0);
      double prev_close=close_prices[1];

      if(prev_close>0)
        {
         double gap_pct=MathAbs((current_open-prev_close)/prev_close)*100.0;

         if(gap_pct>=0.3)
           {
            m_market_context.veto_active=true;
            int size=ArraySize(m_market_context.reasons);
            ArrayResize(m_market_context.reasons,size+1);
            m_market_context.reasons[size]=StringFormat("Price Gap Detected (%.2f%%)",gap_pct);
           }
        }
     }

   m_market_context.last_check=TimeCurrent();

//--- Debug output
   if(InpShowDebug)
     {
      if(m_market_context.veto_active)
        {
         Print("[VETO] MARKET CONTEXT VETO ACTIVE:");
         for(int i=0; i<ArraySize(m_market_context.reasons); i++)
           {
            Print("  - ",m_market_context.reasons[i]);
           }
        }
      else
        {
         Print("[MKT] Market Context: NORMAL");
         Print("  Volatility Ratio: ",DoubleToString(m_market_context.volatility_ratio,2),"x");
         Print("  Max Candle Change: ",DoubleToString(m_market_context.max_candle_change,2),"%");
        }
     }
  }

//+------------------------------------------------------------------+
//| Load CSV backtest data                                            |
//+------------------------------------------------------------------+
bool CGGTHExpert::LoadCSVBacktestData()
  {
   Print("Loading CSV backtest data...");

   bool success=true;

//--- Load 1H data
   if(!LoadCSVLookupFile(PERIOD_H1))
     {
      Print("Warning: Failed to load 1H CSV data");
      success=false;
     }

//--- Load 4H data
   if(!LoadCSVLookupFile(PERIOD_H4))
     {
      Print("Warning: Failed to load 4H CSV data");
      success=false;
     }

//--- Load 1D data
   if(!LoadCSVLookupFile(PERIOD_D1))
     {
      Print("Warning: Failed to load 1D CSV data");
      success=false;
     }

   if(success)
      Print("[INIT] CSV backtest data loaded");

   return success;
  }

bool CGGTHExpert::LoadCSVLookupFile(ENUM_TIMEFRAMES timeframe)
  {
   string tf_str="";
   CCSVPrediction temp_array[];
   int count=0;

   switch(timeframe)
     {
      case PERIOD_H1:
         tf_str="1H";
         break;
      case PERIOD_H4:
         tf_str="4H";
         break;
      case PERIOD_D1:
         tf_str="1D";
         break;
      default:
         return(false);
     }

   string filename=m_symbol+"_"+tf_str+"_lookup.csv";

//--- Try to open from Common folder first
   int file_handle=FileOpen(filename,FILE_READ|FILE_TXT|FILE_ANSI|FILE_COMMON);
   if(file_handle==INVALID_HANDLE)
     {
      file_handle=FileOpen(filename,FILE_READ|FILE_TXT|FILE_ANSI);
      if(file_handle==INVALID_HANDLE)
        {
         Print("ERROR: Cannot open ",filename);
         return(false);
        }
     }

//--- Read header line
   string header_line="";
   while(!FileIsEnding(file_handle))
     {
      header_line=FileReadString(file_handle);
      if(header_line!="") break;
     }

   bool has_full_format=
      (StringFind(header_line,"change_pct")>=0 &&
       StringFind(header_line,"ensemble_std")>=0);

//--- Read all records
   ArrayResize(temp_array,10000);
   double last_price=0;

   while(!FileIsEnding(file_handle))
     {
      string line=FileReadString(file_handle);
      if(line=="" || StringLen(line)<5) continue;

      string parts[];
      int num_parts=StringSplit(line,',',parts);

      if(num_parts<2) continue;

//--- Parse timestamp
      string timestamp_str=parts[0];
      StringTrimLeft(timestamp_str);
      StringTrimRight(timestamp_str);
      StringReplace(timestamp_str,".","-");
      datetime dt=StringToTime(timestamp_str);

//--- Parse prediction
      double prediction=StringToDouble(parts[1]);

      double change_pct=0;
      double ensemble_std=0.025;

//--- If full format, read additional columns
      if(has_full_format && num_parts>=4)
        {
         change_pct=StringToDouble(parts[2]);
         ensemble_std=StringToDouble(parts[3]);
        }
      else
        {
         if(last_price>0)
            change_pct=((prediction-last_price)/last_price)*100.0;
         else
            change_pct=0.0;
        }

      if(dt>0 && prediction>0)
        {
         if(count>=ArraySize(temp_array))
            ArrayResize(temp_array,count+1000);

         temp_array[count].timestamp=dt;
         temp_array[count].prediction=prediction;
         temp_array[count].change_pct=change_pct;
         temp_array[count].ensemble_std=ensemble_std;
         count++;

         last_price=prediction;
        }
     }

   FileClose(file_handle);

   if(count==0)
      return(false);

//--- Store in appropriate member array
   switch(timeframe)
     {
      case PERIOD_H1:
         ArrayResize(m_csv_1H,count);
         ArrayCopy(m_csv_1H,temp_array,0,0,count);
         m_csv_1H_count=count;
         break;

      case PERIOD_H4:
         ArrayResize(m_csv_4H,count);
         ArrayCopy(m_csv_4H,temp_array,0,0,count);
         m_csv_4H_count=count;
         break;

      case PERIOD_D1:
         ArrayResize(m_csv_1D,count);
         ArrayCopy(m_csv_1D,temp_array,0,0,count);
         m_csv_1D_count=count;
         break;
     }

   return(true);
  }

//+------------------------------------------------------------------+
//| Load predictions from CSV for backtesting                         |
//+------------------------------------------------------------------+
bool CGGTHExpert::LoadPredictionsFromCSV()
  {
   datetime current_time=iTime(m_symbol,InpTradingTimeframe,0);

//--- Search 1H predictions
   for(int i=0; i<m_csv_1H_count; i++)
     {
      if(m_csv_1H[i].timestamp==current_time)
        {
         m_pred_1H.prediction=m_csv_1H[i].prediction;
         m_pred_1H.change_pct=m_csv_1H[i].change_pct;
         m_pred_1H.ensemble_std=m_csv_1H[i].ensemble_std;
         m_pred_1H.last_update=current_time;
         m_pred_1H.trade_allowed=true;
         break;
        }
     }

//--- Search 4H predictions
   for(int i=0; i<m_csv_4H_count; i++)
     {
      if(m_csv_4H[i].timestamp==current_time)
        {
         m_pred_4H.prediction=m_csv_4H[i].prediction;
         m_pred_4H.change_pct=m_csv_4H[i].change_pct;
         m_pred_4H.ensemble_std=m_csv_4H[i].ensemble_std;
         m_pred_4H.last_update=current_time;
         m_pred_4H.trade_allowed=true;
         break;
        }
     }

//--- Search 1D predictions
   for(int i=0; i<m_csv_1D_count; i++)
     {
      if(m_csv_1D[i].timestamp==current_time)
        {
         m_pred_1D.prediction=m_csv_1D[i].prediction;
         m_pred_1D.change_pct=m_csv_1D[i].change_pct;
         m_pred_1D.ensemble_std=m_csv_1D[i].ensemble_std;
         m_pred_1D.last_update=current_time;
         m_pred_1D.trade_allowed=true;
         break;
        }
     }

   return(true);
  }

//+------------------------------------------------------------------+
//| Load predictions from JSON for live trading                       |
//+------------------------------------------------------------------+
bool CGGTHExpert::LoadPredictionsFromJSON()
  {
   string filename=m_symbol+"_predictions_multitf.json";
   int file_handle=FileOpen(filename,FILE_READ|FILE_TXT|FILE_ANSI);

   if(file_handle==INVALID_HANDLE)
      return(false);

//--- Read entire file
   string json_content="";
   while(!FileIsEnding(file_handle))
     {
      json_content+=FileReadString(file_handle);
     }
   FileClose(file_handle);

   if(json_content=="")
      return(false);

//--- Parse JSON
   bool success=ParsePredictionJSON(json_content,"1H",m_pred_1H);
   success=success && ParsePredictionJSON(json_content,"4H",m_pred_4H);
   success=success && ParsePredictionJSON(json_content,"1D",m_pred_1D);

   return success;
  }

//+------------------------------------------------------------------+
//| Parse prediction JSON                                             |
//+------------------------------------------------------------------+
bool CGGTHExpert::ParsePredictionJSON(string json,string timeframe,CPredictionData &pred)
  {
   string search_key="\""+timeframe+"\":";
   int pos=StringFind(json,search_key);

   if(pos<0)
      return(false);

//--- Extract prediction value
   string prediction_key="\"prediction\":";
   int pred_pos=StringFind(json,prediction_key,pos);
   if(pred_pos>0)
     {
      int value_start=pred_pos+StringLen(prediction_key);
      int value_end=StringFind(json,",",value_start);
      if(value_end<0)
         value_end=StringFind(json,"}",value_start);

      string value_str=StringSubstr(json,value_start,value_end-value_start);
      StringTrimLeft(value_str);
      StringTrimRight(value_str);
      pred.prediction=StringToDouble(value_str);
     }

//--- Extract change_pct
   string change_key="\"change_pct\":";
   int change_pos=StringFind(json,change_key,pos);
   if(change_pos>0)
     {
      int value_start=change_pos+StringLen(change_key);
      int value_end=StringFind(json,",",value_start);
      if(value_end<0)
         value_end=StringFind(json,"}",value_start);

      string value_str=StringSubstr(json,value_start,value_end-value_start);
      StringTrimLeft(value_str);
      StringTrimRight(value_str);
      pred.change_pct=StringToDouble(value_str);
     }

//--- Extract ensemble_std
   string std_key="\"ensemble_std\":";
   int std_pos=StringFind(json,std_key,pos);
   if(std_pos>0)
     {
      int value_start=std_pos+StringLen(std_key);
      int value_end=StringFind(json,",",value_start);
      if(value_end<0)
         value_end=StringFind(json,"}",value_start);

      string value_str=StringSubstr(json,value_start,value_end-value_start);
      StringTrimLeft(value_str);
      StringTrimRight(value_str);
      pred.ensemble_std=StringToDouble(value_str);
     }

   pred.last_update=TimeCurrent();
   pred.trade_allowed=true;

   return(pred.prediction>0);
  }

//+------------------------------------------------------------------+
//| Update accuracy tracking                                          |
//+------------------------------------------------------------------+
void CGGTHExpert::UpdateAccuracyTracking()
  {
   CheckAccuracyForTimeframe(m_tracker_1H,m_pred_1H,PERIOD_H1);
   CheckAccuracyForTimeframe(m_tracker_4H,m_pred_4H,PERIOD_H4);
   CheckAccuracyForTimeframe(m_tracker_1D,m_pred_1D,PERIOD_D1);
  }

//+------------------------------------------------------------------+
//| Check accuracy for specific timeframe                             |
//+------------------------------------------------------------------+
void CGGTHExpert::CheckAccuracyForTimeframe(CAccuracyTracker &tracker,CPredictionData &pred,ENUM_TIMEFRAMES tf)
  {
//--- If we have a new prediction, record it
   if(pred.last_update>tracker.current_prediction.timestamp && pred.prediction>0)
     {
      if(!tracker.current_prediction.checked)
        {
//--- Check previous prediction if it exists
         if(tracker.current_prediction.timestamp>0)
           {
            datetime check_time=tracker.current_prediction.timestamp;
            int shift=iBarShift(m_symbol,tf,check_time);

            if(shift>=1)
              {
               double actual_price=iClose(m_symbol,tf,shift-1);
               double predicted_direction=(tracker.current_prediction.predicted_price-tracker.current_prediction.start_price);
               double actual_direction=(actual_price-tracker.current_prediction.start_price);

               bool accurate=(predicted_direction*actual_direction>0);

               tracker.total_predictions++;
               if(accurate)
                  tracker.accurate_predictions++;

               if(tracker.total_predictions>0)
                  tracker.accuracy_percent=(double)tracker.accurate_predictions/tracker.total_predictions*100.0;

               tracker.current_prediction.checked=true;
               tracker.current_prediction.accurate=accurate;
              }
           }
        }

//--- Record new prediction
      tracker.current_prediction.timestamp=pred.last_update;
      tracker.current_prediction.predicted_price=pred.prediction;
      tracker.current_prediction.start_price=m_current_price;
      tracker.current_prediction.checked=false;
     }
  }

//+------------------------------------------------------------------+
//| Check for trade signal                                            |
//+------------------------------------------------------------------+
void CGGTHExpert::CheckForTradeSignal()
  {
   if(!InpEnableTrading)
      return;

//--- Check if trading is allowed (day/session filters)
   if(!IsTradingAllowed())
      return;

//--- Apply market context veto early if active
   if(InpUseMarketContextVeto && m_market_context.veto_active)
     {
      if(InpShowDebug)
         Print("Trade blocked by Market Context Veto");
      return;
     }

//--- Check margin utilization - block new positions if margin usage exceeds threshold
   double margin_used  = AccountInfoDouble(ACCOUNT_MARGIN);
   double equity       = AccountInfoDouble(ACCOUNT_EQUITY);
   double margin_usage = (equity > 0) ? (margin_used / equity * 100.0) : 0.0;
   if(margin_usage > InpMaxMarginUsagePct)
     {
      if(InpShowDebug)
         Print("[MARGIN] New trade blocked: margin usage ",DoubleToString(margin_usage,1),
               "% exceeds limit of ",DoubleToString(InpMaxMarginUsagePct,1),"%");
      return;
     }

//--- Get the prediction for selected timeframe
   CPredictionData selected_pred;
   string tf_name="";

   switch(InpTradingTimeframe)
     {
      case PERIOD_H1:
         selected_pred=m_pred_1H;
         tf_name="1H";
         break;
      case PERIOD_H4:
         selected_pred=m_pred_4H;
         tf_name="4H";
         break;
      case PERIOD_D1:
         selected_pred=m_pred_1D;
         tf_name="1D";
         break;
      default:
         Print("ERROR: Unsupported trading timeframe");
         return;
     }

   if(selected_pred.prediction<=0)
      return;

//--- Determine pip size
   double point=SymbolInfoDouble(m_symbol,SYMBOL_POINT);
   double pip=point;
   if(_Digits==3 || _Digits==5)
      pip=point*10.0;

//--- Compute prediction distance in pips
   double delta_pips=(selected_pred.prediction-m_current_price)/pip;

   bool signal_buy=false;
   bool signal_sell=false;

//--- Use adaptive min_pred_pips if learning is enabled, else use input
   double effective_min_pred_pips=InpEnableAdaptiveLearning
                                  ? m_adaptive.min_pred_pips
                                  : (double)InpMinPredictionPips;

   if(effective_min_pred_pips<=0)
     {
      signal_buy=(selected_pred.prediction>m_current_price);
      signal_sell=(selected_pred.prediction<m_current_price);
     }
   else
     {
      signal_buy=(delta_pips>=effective_min_pred_pips);
      signal_sell=(delta_pips<=-effective_min_pred_pips);
     }

//--- If no valid direction, exit
   if(!signal_buy && !signal_sell)
      return;

//--- Apply trend filter
   if(InpUseTrendFilter)
     {
      if(!CheckTrendFilter(signal_buy,signal_sell))
        {
         if(InpShowDebug)
            Print("Trade rejected by trend filter");
         return;
        }
     }

//--- Apply RSI filter
   if(InpUseRSIFilter)
     {
      if(!CheckRSIFilter(signal_buy,signal_sell))
        {
         if(InpShowDebug)
            Print("Trade rejected by RSI filter");
         return;
        }
     }

//--- Calculate position size (apply adaptive lot multiplier)
   double lot_size=CalculateLotSize();
   if(InpEnableAdaptiveLearning)
      lot_size*=m_adaptive.lot_multiplier;
   if(lot_size<=0)
      return;

//--- Re-normalize after multiplier
   {
    double min_lot=SymbolInfoDouble(m_symbol,SYMBOL_VOLUME_MIN);
    double max_lot=SymbolInfoDouble(m_symbol,SYMBOL_VOLUME_MAX);
    double lot_step=SymbolInfoDouble(m_symbol,SYMBOL_VOLUME_STEP);
    lot_size=MathFloor(lot_size/lot_step)*lot_step;
    lot_size=MathMax(lot_size,min_lot);
    lot_size=MathMin(lot_size,max_lot);
   }

//--- FIFO COMPLIANCE: block any trade that would hedge an existing position.
//--- NFA Rule 2-43(b) prohibits opening a position opposite to an existing one
//--- on the same symbol. We must also refuse to open a new position in any
//--- direction while an opposite position is open.
   if(InpFIFOCompliant)
     {
      for(int i=PositionsTotal()-1; i>=0; i--)
        {
         ulong chk_ticket=PositionGetTicket(i);
         if(chk_ticket<=0) continue;
         if(PositionGetString(POSITION_SYMBOL)!=m_symbol) continue;
         if(PositionGetInteger(POSITION_MAGIC)!=InpMagicNumber) continue;

         long existing_type=PositionGetInteger(POSITION_TYPE);
         if(signal_buy && existing_type==POSITION_TYPE_SELL)
           {
            if(InpShowDebug)
               Print("[FIFO] BUY blocked - open SELL position exists (FIFO mode)");
            return;
           }
         if(signal_sell && existing_type==POSITION_TYPE_BUY)
           {
            if(InpShowDebug)
               Print("[FIFO] SELL blocked - open BUY position exists (FIFO mode)");
            return;
           }
        }
     }

//--- Read current RSI for entry record
   double rsi_entry=50.0;
   if(InpUseRSIFilter && m_handle_rsi!=INVALID_HANDLE)
     {
      double rsi_buf[];
      ArraySetAsSeries(rsi_buf,true);
      if(CopyBuffer(m_handle_rsi,0,0,1,rsi_buf)==1)
         rsi_entry=rsi_buf[0];
     }

//--- Calculate SL and TP
   double sl_distance=InpStopLossPips*pip;
   double tp_price=0;
   double tp_distance=0;

//--- Use predicted price as TP
   if(InpUsePredictedPrice)
     {
      tp_price=selected_pred.prediction*InpTPMultiplier;

      double tp_pips=MathAbs(tp_price-m_current_price)/pip;

      if(tp_pips<InpMinTPPips)
         return;

      if(tp_pips>InpMaxTPPips)
        {
         if(signal_buy)
            tp_price=m_current_price+(InpMaxTPPips*pip);
         else
            tp_price=m_current_price-(InpMaxTPPips*pip);
        }
     }
   else
     {
      tp_distance=InpTakeProfitPips*pip;
     }

//--- Execute trade
   if(signal_buy)
     {
      double ask=SymbolInfoDouble(m_symbol,SYMBOL_ASK);
      double sl=ask-sl_distance;
      double tp=InpUsePredictedPrice ? tp_price : ask+tp_distance;

      if(tp<=ask)
         return;

      string comment=StringFormat("GGTH v1.06 [%s] %s%.2f%% TP:%."+IntegerToString(_Digits)+"f",
                                  tf_name,
                                  (selected_pred.change_pct>=0 ? "+" : ""),
                                  selected_pred.change_pct,
                                  tp);

      if(m_trade.Buy(lot_size,m_symbol,ask,sl,tp,comment))
        {
         m_last_trade_time=TimeCurrent();
         //--- Record entry for adaptive learning
         //--- In MT5, position_id == the opening ORDER ticket (ResultOrder),
         //--- which matches DEAL_POSITION_ID on the closing deal.
         if(InpEnableAdaptiveLearning)
           {
            ulong pos_id=m_trade.ResultOrder();
            RecordTradeEntry(pos_id,true,selected_pred.change_pct,MathAbs(delta_pips),rsi_entry);
           }
         Print("GGTH BUY placed [",tf_name,"]",
               " pred:",DoubleToString(selected_pred.prediction,_Digits),
               " TP:",DoubleToString(tp,_Digits),
               " SL:",DoubleToString(sl,_Digits),
               " LotMult:",DoubleToString(m_adaptive.lot_multiplier,2));
        }
     }
   else if(signal_sell)
     {
      double bid=SymbolInfoDouble(m_symbol,SYMBOL_BID);
      double sl=bid+sl_distance;
      double tp=InpUsePredictedPrice ? tp_price : bid-tp_distance;

      if(tp>=bid)
         return;

      string comment=StringFormat("GGTH v1.06 [%s] %.2f%% TP:%."+IntegerToString(_Digits)+"f",
                                  tf_name,
                                  selected_pred.change_pct,
                                  tp);

      if(m_trade.Sell(lot_size,m_symbol,bid,sl,tp,comment))
        {
         m_last_trade_time=TimeCurrent();
         //--- Record entry for adaptive learning
         if(InpEnableAdaptiveLearning)
           {
            ulong pos_id=m_trade.ResultOrder();
            RecordTradeEntry(pos_id,false,selected_pred.change_pct,MathAbs(delta_pips),rsi_entry);
           }
         Print("GGTH SELL placed [",tf_name,"]",
               " pred:",DoubleToString(selected_pred.prediction,_Digits),
               " TP:",DoubleToString(tp,_Digits),
               " SL:",DoubleToString(sl,_Digits),
               " LotMult:",DoubleToString(m_adaptive.lot_multiplier,2));
        }
     }
  }

//+------------------------------------------------------------------+
//| Check trend filter                                                |
//+------------------------------------------------------------------+
bool CGGTHExpert::CheckTrendFilter(bool &signal_buy,bool &signal_sell)
  {
   double ma_buffer[];
   ArraySetAsSeries(ma_buffer,true);

   if(CopyBuffer(m_handle_trend_ma,0,0,2,ma_buffer)!=2)
      return(false);

   double current_ma=ma_buffer[0];
   double current_close=iClose(m_symbol,Period(),0);

//--- Only allow buy if price is above MA
   if(signal_buy && current_close<current_ma)
     {
      signal_buy=false;
      return(false);
     }

//--- Only allow sell if price is below MA
   if(signal_sell && current_close>current_ma)
     {
      signal_sell=false;
      return(false);
     }

   return(true);
  }

//+------------------------------------------------------------------+
//| Check RSI filter                                                  |
//+------------------------------------------------------------------+
bool CGGTHExpert::CheckRSIFilter(bool &signal_buy,bool &signal_sell)
  {
   double rsi_buffer[];
   ArraySetAsSeries(rsi_buffer,true);

   if(CopyBuffer(m_handle_rsi,0,0,2,rsi_buffer)!=2)
      return(false);

   double current_rsi=rsi_buffer[0];

//--- Use adaptive RSI levels if learning enabled
   double ob_level=InpEnableAdaptiveLearning ? m_adaptive.rsi_overbought : InpRSIOverbought;
   double os_level=InpEnableAdaptiveLearning ? m_adaptive.rsi_oversold   : InpRSIOversold;

//--- Don't buy if RSI is overbought
   if(signal_buy && current_rsi>ob_level)
     {
      signal_buy=false;
      return(false);
     }

//--- Don't sell if RSI is oversold
   if(signal_sell && current_rsi<os_level)
     {
      signal_sell=false;
      return(false);
     }

   return(true);
  }

//+------------------------------------------------------------------+
//| Check if trading is allowed                                       |
//+------------------------------------------------------------------+
bool CGGTHExpert::IsTradingAllowed()
  {
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(),dt);

//--- Check day of week
   switch(dt.day_of_week)
     {
      case 1:
         if(!InpTradeMonday) return(false);
         break;
      case 2:
         if(!InpTradeTuesday) return(false);
         break;
      case 3:
         if(!InpTradeWednesday) return(false);
         break;
      case 4:
         if(!InpTradeThursday) return(false);
         break;
      case 5:
         if(!InpTradeFriday) return(false);
         break;
      case 6:
         if(!InpTradeSaturday) return(false);
         break;
      case 0:
         if(!InpTradeSunday) return(false);
         break;
     }

//--- Check trading sessions
   return IsWithinTradingSession(dt.hour,dt.min);
  }

//+------------------------------------------------------------------+
//| Check if within trading session                                   |
//+------------------------------------------------------------------+
bool CGGTHExpert::IsWithinTradingSession(int hour,int minute)
  {
   int current_minutes=hour*60+minute;

//--- Check Session 1
   if(InpUseSession1)
     {
      int start1=InpSession1StartHour*60+InpSession1StartMinute;
      int end1=InpSession1EndHour*60+InpSession1EndMinute;

      if(start1<=end1)
        {
         if(current_minutes>=start1 && current_minutes<=end1)
            return(true);
        }
      else
        {
         if(current_minutes>=start1 || current_minutes<=end1)
            return(true);
        }
     }

//--- Check Session 2
   if(InpUseSession2)
     {
      int start2=InpSession2StartHour*60+InpSession2StartMinute;
      int end2=InpSession2EndHour*60+InpSession2EndMinute;

      if(start2<=end2)
        {
         if(current_minutes>=start2 && current_minutes<=end2)
            return(true);
        }
      else
        {
         if(current_minutes>=start2 || current_minutes<=end2)
            return(true);
        }
     }

//--- Check Session 3
   if(InpUseSession3)
     {
      int start3=InpSession3StartHour*60+InpSession3StartMinute;
      int end3=InpSession3EndHour*60+InpSession3EndMinute;

      if(start3<=end3)
        {
         if(current_minutes>=start3 && current_minutes<=end3)
            return(true);
        }
      else
        {
         if(current_minutes>=start3 || current_minutes<=end3)
            return(true);
        }
     }

   return(false);
  }

//+------------------------------------------------------------------+
//| Display information - MAXIMUM SPACING VERSION                     |
//+------------------------------------------------------------------+
void CGGTHExpert::DisplayInfo()
  {
   int x_pos=InpXOffset;
   int y_pos=InpYOffset;
   int line_height=InpFontSize+20;

//--- Header
   CreateLabel("MLEA_Header",x_pos,y_pos,
               "╔══════════════════════════════════════════════════════════════╗",
               InpFontSize,InpTextColor);
   y_pos+=line_height;

   CreateLabel("MLEA_Title",x_pos,y_pos,
               "║              GGTH ML PREDICTOR v1.06                       ║",
               InpFontSize+3,clrGold);
   y_pos+=line_height+10;

//--- Current price
   string price_text=
      StringFormat("║     %s:  %."+IntegerToString(_Digits)+"f",
                   m_symbol,m_current_price);
   while(StringLen(price_text)<62) price_text+=" ";
   price_text+="║";
   CreateLabel("MLEA_Price",x_pos,y_pos,price_text,InpFontSize+2,clrWhite);
   y_pos+=line_height+15;

//--- Separator
   CreateLabel("MLEA_Separator1",x_pos,y_pos,
               "╠══════════════════════════════════════════════════════════════╣",
               InpFontSize,InpTextColor);
   y_pos+=line_height+10;

//--- Predictions header
   CreateLabel("MLEA_PredHeader",x_pos,y_pos,
               "║     PREDICTIONS:                                           ║",
               InpFontSize+1,clrYellow);
   y_pos+=line_height+15;

//--- Display predictions with HUGE spacing
   DisplayPredictionLine("1H",m_pred_1H,m_tracker_1H,x_pos,y_pos);
   y_pos+=line_height*3+20;

   DisplayPredictionLine("4H",m_pred_4H,m_tracker_4H,x_pos,y_pos);
   y_pos+=line_height*3+20;

   DisplayPredictionLine("1D",m_pred_1D,m_tracker_1D,x_pos,y_pos);
   y_pos+=line_height*2+15;

//--- Adaptive Learning section
   if(InpEnableAdaptiveLearning)
      DisplayAdaptiveInfo(x_pos,y_pos,line_height);

//--- Footer
   CreateLabel("MLEA_Footer",x_pos,y_pos,
               "╚══════════════════════════════════════════════════════════════╝",
               InpFontSize,InpTextColor);

   ChartRedraw();
  }

//+------------------------------------------------------------------+
//| Display prediction line - MAXIMUM SPACING VERSION                 |
//+------------------------------------------------------------------+
void CGGTHExpert::DisplayPredictionLine(string tf_name,CPredictionData &pred,
                                        CAccuracyTracker &tracker,int x_pos,int &y_pos)
  {
   color line_color=(pred.prediction>m_current_price) ? InpUpColor : InpDownColor;
   string arrow=(pred.prediction>m_current_price) ? "UP" : "DN";
   string direction=(pred.prediction>m_current_price) ? "UP" : "DOWN";

//--- Main prediction line
   string pred_text=
      StringFormat("║     %s  %s %s   %."+IntegerToString(_Digits)+"f   (%s%.2f%%)",
                   tf_name,direction,arrow,pred.prediction,
                   (pred.change_pct>=0?"+":""),pred.change_pct);
   while(StringLen(pred_text)<62) pred_text+=" ";
   pred_text+="║";

   CreateLabel("MLEA_Pred_"+tf_name,x_pos,y_pos,pred_text,
               InpFontSize+2,line_color);
   y_pos+=40;

//--- Accuracy line
   string accuracy_text=
      StringFormat("║          Accuracy: %d / %d   (%.1f%%)                    ",
                   tracker.accurate_predictions,
                   tracker.total_predictions,
                   tracker.accuracy_percent);
   while(StringLen(accuracy_text)<62) accuracy_text+=" ";
   accuracy_text+="║";

   color acc_color=(tracker.accuracy_percent>=60) ? clrLimeGreen :
                   (tracker.accuracy_percent>=50) ? clrYellow : clrRed;

   CreateLabel("MLEA_Acc_"+tf_name,x_pos,y_pos,accuracy_text,
               InpFontSize,acc_color);
  }

//+------------------------------------------------------------------+
//| Display error message                                             |
//+------------------------------------------------------------------+
void CGGTHExpert::DisplayError()
  {
   int x_pos=InpXOffset;
   int y_pos=InpYOffset;
   int line_height=InpFontSize+8;

   CreateLabel("MLEA_Error1",x_pos,y_pos,
               "[!] Waiting for ML predictions...",
               InpFontSize+2,clrOrange);
   y_pos+=line_height+5;

   CreateLabel("MLEA_Error2",x_pos,y_pos,
               "Run: python predictor.py predict --symbol "+m_symbol,
               InpFontSize-1,clrGray);

   ChartRedraw();
  }

//+------------------------------------------------------------------+
//| Create label helper                                               |
//+------------------------------------------------------------------+
void CGGTHExpert::CreateLabel(string name,int x,int y,string text,
                               int font_size,color clr)
  {
   if(ObjectFind(0,name)<0)
     {
      ObjectCreate(0,name,OBJ_LABEL,0,0,0);
      ObjectSetInteger(0,name,OBJPROP_CORNER,CORNER_LEFT_UPPER);
      ObjectSetInteger(0,name,OBJPROP_ANCHOR,ANCHOR_LEFT_UPPER);
     }

   ObjectSetInteger(0,name,OBJPROP_XDISTANCE,x);
   ObjectSetInteger(0,name,OBJPROP_YDISTANCE,y);
   ObjectSetString(0,name,OBJPROP_TEXT,text);
   ObjectSetInteger(0,name,OBJPROP_FONTSIZE,font_size);
   ObjectSetInteger(0,name,OBJPROP_COLOR,clr);
   ObjectSetString(0,name,OBJPROP_FONT,"Courier New");
  }

//+------------------------------------------------------------------+
//| Save accuracy data                                                |
//+------------------------------------------------------------------+
void CGGTHExpert::SaveAccuracyData()
  {
   string filename="accuracy_"+m_symbol+"_v8.dat";
   int handle=FileOpen(filename,FILE_WRITE|FILE_BIN);

   if(handle!=INVALID_HANDLE)
     {
      FileWriteInteger(handle,m_tracker_1H.total_predictions);
      FileWriteInteger(handle,m_tracker_1H.accurate_predictions);
      FileWriteInteger(handle,m_tracker_4H.total_predictions);
      FileWriteInteger(handle,m_tracker_4H.accurate_predictions);
      FileWriteInteger(handle,m_tracker_1D.total_predictions);
      FileWriteInteger(handle,m_tracker_1D.accurate_predictions);
      FileClose(handle);

      if(InpShowDebug)
         Print("[SAVE] Accuracy data saved");
     }
  }

//+------------------------------------------------------------------+
//| Load accuracy data                                                |
//+------------------------------------------------------------------+
void CGGTHExpert::LoadAccuracyData()
  {
   string filename="accuracy_"+m_symbol+"_v8.dat";
   int handle=FileOpen(filename,FILE_READ|FILE_BIN);

   if(handle!=INVALID_HANDLE)
     {
      m_tracker_1H.total_predictions=FileReadInteger(handle);
      m_tracker_1H.accurate_predictions=FileReadInteger(handle);
      if(m_tracker_1H.total_predictions>0)
         m_tracker_1H.accuracy_percent=
            (double)m_tracker_1H.accurate_predictions/
            m_tracker_1H.total_predictions*100.0;

      m_tracker_4H.total_predictions=FileReadInteger(handle);
      m_tracker_4H.accurate_predictions=FileReadInteger(handle);
      if(m_tracker_4H.total_predictions>0)
         m_tracker_4H.accuracy_percent=
            (double)m_tracker_4H.accurate_predictions/
            m_tracker_4H.total_predictions*100.0;

      m_tracker_1D.total_predictions=FileReadInteger(handle);
      m_tracker_1D.accurate_predictions=FileReadInteger(handle);
      if(m_tracker_1D.total_predictions>0)
         m_tracker_1D.accuracy_percent=
            (double)m_tracker_1D.accurate_predictions/
            m_tracker_1D.total_predictions*100.0;

      FileClose(handle);
      Print("[LOAD] Historical accuracy data loaded");
     }
  }

//+------------------------------------------------------------------+
//| Initialize adaptive state to defaults from inputs                 |
//+------------------------------------------------------------------+
void CGGTHExpert::InitAdaptiveState()
  {
   m_adaptive.min_pred_pips     = (double)InpMinPredictionPips;
   m_adaptive.lot_multiplier    = 1.0;
   m_adaptive.rsi_overbought    = InpRSIOverbought;
   m_adaptive.rsi_oversold      = InpRSIOversold;
   m_adaptive.win_rate          = 0.5;
   m_adaptive.profit_factor     = 1.0;
   m_adaptive.avg_win           = 0.0;
   m_adaptive.avg_loss          = 0.0;
   m_adaptive.kelly_fraction    = 0.0;
   m_adaptive.trades_since_last_adapt = 0;
   m_adaptive.total_adaptations = 0;
   m_adaptive.consecutive_losses= 0;
   m_adaptive.history_head      = 0;
   m_adaptive.history_size      = 0;

   for(int i=0; i<MAX_OPEN_TRACKING; i++)
      m_open_tracking[i].used=false;

   Print("[INIT] Adaptive learning state initialised (defaults from inputs)");
  }

//+------------------------------------------------------------------+
//| Record conditions at trade entry                                   |
//+------------------------------------------------------------------+
void CGGTHExpert::RecordTradeEntry(ulong position_id,bool is_buy,
                                    double pred_change_pct,double pred_pips,
                                    double rsi)
  {
   if(!InpEnableAdaptiveLearning || position_id==0) return;

//--- Find free slot
   for(int i=0; i<MAX_OPEN_TRACKING; i++)
     {
      if(!m_open_tracking[i].used)
        {
         m_open_tracking[i].position_id     = position_id;
         m_open_tracking[i].pred_change_pct = pred_change_pct;
         m_open_tracking[i].pred_pips       = pred_pips;
         m_open_tracking[i].rsi_at_entry    = rsi;
         m_open_tracking[i].was_buy         = is_buy;
         m_open_tracking[i].used            = true;
         if(InpShowAdaptiveDebug)
            Print("[ENTRY] Recorded entry for position_id ",position_id,
                  " | predPips:",DoubleToString(pred_pips,1),
                  " | RSI:",DoubleToString(rsi,1));
         return;
        }
     }
   Print("[WARN] Adaptive: open tracking array full - entry not recorded");
  }

//+------------------------------------------------------------------+
//| Called from OnTradeTransaction when a position closes             |
//+------------------------------------------------------------------+
void CGGTHExpert::ProcessClosedTrade(ulong position_id,double profit)
  {
   if(!InpEnableAdaptiveLearning) return;

//--- Find matching entry in open tracking
   bool found=false;
   COpenTradeEntry entry;
   ZeroMemory(entry);

   for(int i=0; i<MAX_OPEN_TRACKING; i++)
     {
      if(m_open_tracking[i].used && m_open_tracking[i].position_id==position_id)
        {
         entry = m_open_tracking[i];
         m_open_tracking[i].used=false;
         found=true;
         break;
        }
     }

   if(!found)
     {
      //--- Trade placed before adaptive learning was active - still count it
      if(InpShowAdaptiveDebug)
         Print("[INFO] Adaptive: no entry record for position_id ",position_id,
               " - counting profit only");
     }

//--- Write to circular history buffer
   int idx = m_adaptive.history_head;
   m_adaptive.history[idx].position_id     = position_id;
   m_adaptive.history[idx].profit          = profit;
   m_adaptive.history[idx].won             = (profit>0);
   m_adaptive.history[idx].close_time      = TimeCurrent();
   m_adaptive.history[idx].used            = true;

   if(found)
     {
      m_adaptive.history[idx].pred_change_pct = entry.pred_change_pct;
      m_adaptive.history[idx].pred_pips       = entry.pred_pips;
      m_adaptive.history[idx].rsi_at_entry    = entry.rsi_at_entry;
      m_adaptive.history[idx].was_buy         = entry.was_buy;
     }

//--- Advance circular head
   m_adaptive.history_head = (m_adaptive.history_head + 1) % MAX_TRADE_HISTORY;
   if(m_adaptive.history_size < MAX_TRADE_HISTORY)
      m_adaptive.history_size++;

//--- Track consecutive losses
   if(profit>0)
      m_adaptive.consecutive_losses=0;
   else
      m_adaptive.consecutive_losses++;

   m_adaptive.trades_since_last_adapt++;

   if(InpShowAdaptiveDebug)
      Print("[TRADE] Closed | P/L: ",DoubleToString(profit,2),
            " | ConsecLoss: ",m_adaptive.consecutive_losses,
            " | TradesSinceLast: ",m_adaptive.trades_since_last_adapt);

//--- Adapt if we've accumulated enough trades
   if(m_adaptive.trades_since_last_adapt >= InpAdaptEveryN)
      AdaptParameters();
  }

//+------------------------------------------------------------------+
//| Core adaptation algorithm                                         |
//+------------------------------------------------------------------+
void CGGTHExpert::AdaptParameters()
  {
   if(m_adaptive.history_size < 3) return; // need at least 3 trades

   double win_rate=0, profit_factor=0, avg_win=0, avg_loss=0;
   ComputeRollingMetrics(win_rate,profit_factor,avg_win,avg_loss);

   m_adaptive.win_rate      = win_rate;
   m_adaptive.profit_factor = profit_factor;
   m_adaptive.avg_win       = avg_win;
   m_adaptive.avg_loss      = avg_loss;

   double kelly = ComputeKellyFraction(win_rate,avg_win,avg_loss);
   m_adaptive.kelly_fraction = kelly;

   double rate = InpAdaptRate;

   //--- 1) Adapt min_pred_pips ----------------------------------------
   //    Tighten (raise) when losing to demand stronger signals.
   //    Loosen (lower) when consistently winning.
   double target_pred_pips = m_adaptive.min_pred_pips;
   if(win_rate < 0.40)
      target_pred_pips = m_adaptive.min_pred_pips * (1.0 + rate);
   else if(win_rate >= 0.60 && profit_factor >= 1.5)
      target_pred_pips = m_adaptive.min_pred_pips * (1.0 - rate * 0.5);

   m_adaptive.min_pred_pips = MathMax(InpAdaptMinPredFloor,
                              MathMin(InpAdaptMinPredCeil, target_pred_pips));

   //--- 2) Adapt lot_multiplier via half-Kelly -----------------------
   //    Kelly fraction clipped to [0, 0.5]; multiplier = kelly * 2
   double kelly_clamped = MathMax(0.0, MathMin(0.5, kelly));
   double kelly_target  = kelly_clamped * 2.0;  // maps [0,0.5] -> [0,1]
   if(kelly_target < 0.25) kelly_target = 0.25; // minimum participation

   // Smooth towards kelly target using learning rate
   m_adaptive.lot_multiplier += rate * (kelly_target - m_adaptive.lot_multiplier);
   m_adaptive.lot_multiplier  = MathMax(InpAdaptLotMultFloor,
                                MathMin(InpAdaptLotMultCeil, m_adaptive.lot_multiplier));

   //--- 3) Adapt RSI levels -------------------------------------------
   //    Tighten (more restrictive) when win rate is poor.
   //    Relax when strong.
   if(win_rate < 0.40)
     {
      // Tighten: move OB down, OS up -> fewer signals pass
      m_adaptive.rsi_overbought = MathMax(55.0, m_adaptive.rsi_overbought - rate * 5.0);
      m_adaptive.rsi_oversold   = MathMin(45.0, m_adaptive.rsi_oversold   + rate * 5.0);
     }
   else if(win_rate >= 0.60 && profit_factor >= 1.5)
     {
      // Relax: widen RSI window
      m_adaptive.rsi_overbought = MathMin(80.0, m_adaptive.rsi_overbought + rate * 3.0);
      m_adaptive.rsi_oversold   = MathMax(20.0, m_adaptive.rsi_oversold   - rate * 3.0);
     }

   m_adaptive.trades_since_last_adapt = 0;
   m_adaptive.total_adaptations++;

   Print("[ADAPT] UPDATE #",m_adaptive.total_adaptations);
   Print("   WinRate: ",DoubleToString(win_rate*100,1),"%",
         " | PF: ",DoubleToString(profit_factor,2),
         " | Kelly: ",DoubleToString(kelly*100,1),"%");
   Print("   MinPredPips: ",DoubleToString(m_adaptive.min_pred_pips,1),
         " | LotMult: ",DoubleToString(m_adaptive.lot_multiplier,3),
         " | RSI OB/OS: ",DoubleToString(m_adaptive.rsi_overbought,1),
         "/",DoubleToString(m_adaptive.rsi_oversold,1));

//--- Save immediately so state is not lost if EA crashes
   SaveAdaptiveState();
  }

//+------------------------------------------------------------------+
//| Compute rolling performance metrics over last InpAdaptLookback   |
//+------------------------------------------------------------------+
void CGGTHExpert::ComputeRollingMetrics(double &win_rate,double &profit_factor,
                                         double &avg_win,double &avg_loss)
  {
   int window = MathMin(InpAdaptLookback, m_adaptive.history_size);
   if(window==0) { win_rate=0.5; profit_factor=1.0; avg_win=0; avg_loss=0; return; }

   int wins=0;
   double gross_win=0, gross_loss=0;

//--- Walk backwards through circular buffer
   for(int k=0; k<window; k++)
     {
      int idx = (m_adaptive.history_head - 1 - k + MAX_TRADE_HISTORY) % MAX_TRADE_HISTORY;
      if(!m_adaptive.history[idx].used) continue;

      double p = m_adaptive.history[idx].profit;
      if(p > 0) { wins++; gross_win  += p; }
      else       {         gross_loss += MathAbs(p); }
     }

   win_rate      = (double)wins / window;
   profit_factor = (gross_loss > 0) ? gross_win / gross_loss : (gross_win > 0 ? 10.0 : 1.0);
   avg_win       = (wins > 0) ? gross_win  / wins         : 0.0;
   avg_loss      = (window - wins > 0) ? gross_loss / (window - wins) : 0.0;
  }

//+------------------------------------------------------------------+
//| Half-Kelly fraction: f = p - (1-p)/b  where b = avg_win/avg_loss |
//+------------------------------------------------------------------+
double CGGTHExpert::ComputeKellyFraction(double win_rate,double avg_win,double avg_loss)
  {
   if(avg_loss <= 0 || avg_win <= 0) return 0.0;
   double b = avg_win / avg_loss;          // payoff ratio
   double f = win_rate - (1.0 - win_rate) / b;
   return MathMax(0.0, f * 0.5);           // half-Kelly for safety
  }

//+------------------------------------------------------------------+
//| OnTradeTransaction - fires on every deal (entry and exit)        |
//+------------------------------------------------------------------+
void CGGTHExpert::OnTradeTransaction(const MqlTradeTransaction &trans,
                                      const MqlTradeRequest &request,
                                      const MqlTradeResult &result)
  {
   if(!InpEnableAdaptiveLearning) return;

//--- We only care about completed deals on our symbol
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
   if(trans.symbol != m_symbol) return;

   ulong deal_ticket = trans.deal;
   if(deal_ticket == 0) return;

//--- Select the deal from history
   if(!HistoryDealSelect(deal_ticket)) return;

   long deal_entry = HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);

//--- Only process closing deals (DEAL_ENTRY_OUT or DEAL_ENTRY_INOUT)
   if(deal_entry != DEAL_ENTRY_OUT && deal_entry != DEAL_ENTRY_INOUT) return;

   double   profit      = HistoryDealGetDouble(deal_ticket,  DEAL_PROFIT)
                        + HistoryDealGetDouble(deal_ticket,  DEAL_SWAP)
                        + HistoryDealGetDouble(deal_ticket,  DEAL_COMMISSION);
   ulong    position_id = HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID);

   ProcessClosedTrade(position_id, profit);
  }

//+------------------------------------------------------------------+
//| Save adaptive state to file                                       |
//+------------------------------------------------------------------+
void CGGTHExpert::SaveAdaptiveState()
  {
   string filename="adaptive_"+m_symbol+"_v1.dat";
   int h=FileOpen(filename,FILE_WRITE|FILE_BIN);
   if(h==INVALID_HANDLE)
     {
      Print("[ERROR] Adaptive: could not open ",filename," for writing");
      return;
     }

//--- Write header version tag
   FileWriteInteger(h,20260001);

//--- Write adapted parameters
   FileWriteDouble(h,m_adaptive.min_pred_pips);
   FileWriteDouble(h,m_adaptive.lot_multiplier);
   FileWriteDouble(h,m_adaptive.rsi_overbought);
   FileWriteDouble(h,m_adaptive.rsi_oversold);

//--- Write metrics
   FileWriteDouble(h,m_adaptive.win_rate);
   FileWriteDouble(h,m_adaptive.profit_factor);
   FileWriteDouble(h,m_adaptive.avg_win);
   FileWriteDouble(h,m_adaptive.avg_loss);
   FileWriteDouble(h,m_adaptive.kelly_fraction);

//--- Write counters
   FileWriteInteger(h,m_adaptive.total_adaptations);
   FileWriteInteger(h,m_adaptive.consecutive_losses);
   FileWriteInteger(h,m_adaptive.history_head);
   FileWriteInteger(h,m_adaptive.history_size);

//--- Write history records
   for(int i=0; i<MAX_TRADE_HISTORY; i++)
     {
      FileWriteLong(h,(long)m_adaptive.history[i].position_id);
      FileWriteLong(h,(long)m_adaptive.history[i].close_time);
      FileWriteDouble(h,m_adaptive.history[i].profit);
      FileWriteDouble(h,m_adaptive.history[i].pred_change_pct);
      FileWriteDouble(h,m_adaptive.history[i].pred_pips);
      FileWriteDouble(h,m_adaptive.history[i].rsi_at_entry);
      FileWriteInteger(h,m_adaptive.history[i].was_buy ? 1 : 0);
      FileWriteInteger(h,m_adaptive.history[i].won     ? 1 : 0);
      FileWriteInteger(h,m_adaptive.history[i].used    ? 1 : 0);
     }

   FileClose(h);
   if(InpShowAdaptiveDebug)
      Print("[SAVE] Adaptive state saved (",m_adaptive.history_size," trades)");
  }

//+------------------------------------------------------------------+
//| Load adaptive state from file                                     |
//+------------------------------------------------------------------+
void CGGTHExpert::LoadAdaptiveState()
  {
   string filename="adaptive_"+m_symbol+"_v1.dat";
   int h=FileOpen(filename,FILE_READ|FILE_BIN);
   if(h==INVALID_HANDLE)
     {
      Print("[INFO] Adaptive: no saved state found - using input defaults");
      return;
     }

   int version=FileReadInteger(h);
   if(version!=20260001)
     {
      FileClose(h);
      Print("[WARN] Adaptive: state file version mismatch - using defaults");
      return;
     }

   m_adaptive.min_pred_pips  = FileReadDouble(h);
   m_adaptive.lot_multiplier = FileReadDouble(h);
   m_adaptive.rsi_overbought = FileReadDouble(h);
   m_adaptive.rsi_oversold   = FileReadDouble(h);
   m_adaptive.win_rate       = FileReadDouble(h);
   m_adaptive.profit_factor  = FileReadDouble(h);
   m_adaptive.avg_win        = FileReadDouble(h);
   m_adaptive.avg_loss       = FileReadDouble(h);
   m_adaptive.kelly_fraction = FileReadDouble(h);
   m_adaptive.total_adaptations  = FileReadInteger(h);
   m_adaptive.consecutive_losses = FileReadInteger(h);
   m_adaptive.history_head   = FileReadInteger(h);
   m_adaptive.history_size   = FileReadInteger(h);

   for(int i=0; i<MAX_TRADE_HISTORY; i++)
     {
      m_adaptive.history[i].position_id     = (ulong)FileReadLong(h);
      m_adaptive.history[i].close_time      = (datetime)FileReadLong(h);
      m_adaptive.history[i].profit          = FileReadDouble(h);
      m_adaptive.history[i].pred_change_pct = FileReadDouble(h);
      m_adaptive.history[i].pred_pips       = FileReadDouble(h);
      m_adaptive.history[i].rsi_at_entry    = FileReadDouble(h);
      m_adaptive.history[i].was_buy         = (FileReadInteger(h)==1);
      m_adaptive.history[i].won             = (FileReadInteger(h)==1);
      m_adaptive.history[i].used            = (FileReadInteger(h)==1);
     }

   FileClose(h);
   Print("[LOAD] Adaptive state loaded: ",m_adaptive.history_size," trades | ",
         m_adaptive.total_adaptations," adaptations");
   Print("  MinPredPips: ",DoubleToString(m_adaptive.min_pred_pips,1),
         " | LotMult: ",DoubleToString(m_adaptive.lot_multiplier,3),
         " | WinRate: ",DoubleToString(m_adaptive.win_rate*100,1),"%");
  }

//+------------------------------------------------------------------+
//| Display adaptive learning state on chart                          |
//+------------------------------------------------------------------+
void CGGTHExpert::DisplayAdaptiveInfo(int x_pos,int &y_pos,int line_height)
  {
//--- Separator
   CreateLabel("MLEA_AdaptSep",x_pos,y_pos,
               "╠══════════════════════════════════════════════════════════════╣",
               InpFontSize,InpTextColor);
   y_pos+=line_height+5;

//--- Header
   CreateLabel("MLEA_AdaptHeader",x_pos,y_pos,
               "║  [AI] ADAPTIVE LEARNING                                      ║",
               InpFontSize+1,clrCyan);
   y_pos+=line_height+8;

//--- Win rate & profit factor
   string wr_text=StringFormat("║     WinRate: %.1f%%   PF: %.2f   Kelly: %.1f%%         ",
                               m_adaptive.win_rate*100,
                               m_adaptive.profit_factor,
                               m_adaptive.kelly_fraction*100);
   while(StringLen(wr_text)<62) wr_text+=" ";
   wr_text+="║";
   color wr_color=(m_adaptive.win_rate>=0.55) ? clrLimeGreen :
                  (m_adaptive.win_rate>=0.40) ? clrYellow : clrRed;
   CreateLabel("MLEA_AdaptWR",x_pos,y_pos,wr_text,InpFontSize,wr_color);
   y_pos+=line_height+5;

//--- Adapted parameters
   string pp_text=StringFormat("║     MinPredPips: %.1f   LotMult: %.3f            ",
                               m_adaptive.min_pred_pips,
                               m_adaptive.lot_multiplier);
   while(StringLen(pp_text)<62) pp_text+=" ";
   pp_text+="║";
   CreateLabel("MLEA_AdaptPP",x_pos,y_pos,pp_text,InpFontSize,clrWhite);
   y_pos+=line_height+5;

//--- RSI levels + trade count
   string rsi_text=StringFormat("║     RSI OB/OS: %.0f/%.0f   Trades: %d   Adapt#: %d   ",
                                m_adaptive.rsi_overbought,
                                m_adaptive.rsi_oversold,
                                m_adaptive.history_size,
                                m_adaptive.total_adaptations);
   while(StringLen(rsi_text)<62) rsi_text+=" ";
   rsi_text+="║";
   CreateLabel("MLEA_AdaptRSI",x_pos,y_pos,rsi_text,InpFontSize,clrWhite);
   y_pos+=line_height+10;
  }
//+------------------------------------------------------------------+



