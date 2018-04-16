class QResult():
    def __init__(self, hashid, query, federatecash, biddings, prices, links, cashlist):
        self.hashid = hashid
        self.query = query
        self.federatecash = federatecash
        self.totalcash = sum(federatecash)
        self.biddings = biddings
        self.prices = prices
        self.links = links
        self.cashlist = cashlist


class DesignClass():
    def __init__(self, id, query):
        self.id = id
        self.query = query
        self.prices = []
        self.pricestd = []
        self.price50 = []
        self.price25 = []
        self.price75 = []
        self.price10 = []
        self.price90 = []
        self.bids = []
        self.bidstd = []
        self.bid50 = []
        self.bid25 = []
        self.bid75 = []
        self.bid10 = []
        self.bid90 = []
        self.totalcash = []
        self.federatecash = []
        self.links = []
        self.avgprices = []
        self.avgbids = []
        self.totallinks = []
        self.cashes = []
        self.totalcashes = []
