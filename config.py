class Config:
    """Configuration class for model parameters and features."""
    def __init__(self):
        self.target_col = 'target next interval ret'
        self.multi_target_cols = []
        self.valid_ratio = 0.05
        self.start_dt = 1100
        self.seed = 42
        self.feature_cols = None  # Will be set after feature creation
        self.symbols = ['BTCUSDT', 'AVAXUSDT', 'SOLUSDT', 'SHIBUSDT', 'BNBUSDT', 
                       'DOGEUSDT', 'ADAUSDT', 'ETHUSDT', 'XRPUSDT', 'TRXUSDT']

    def set_feature_cols(self, df):
        """Set feature columns excluding target and time columns."""
        self.feature_cols = [col for col in df.columns 
                           if col not in [self.target_col, 'Open time', 'Close time', 'Unnamed: 0']]