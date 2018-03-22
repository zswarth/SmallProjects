## Library to Read Honeywell Thermostat


#This is my first time trying to interact with a physical device.
#I used code by Brad Goodman to make a library I can use later on to interact with my device
	# By Brad Goodman
	# http://www.bradgoodman.com/
	# brad@bradgoodman.com
# You can find his origional code on his website



#Set peramters to log into my personal thermostat
# USERNAME = 'zachary.swarth@gmail.com'
# PASSWORD = 'Zachary1'
# DEVICE_ID = 699427

# AUTH="https://mytotalconnectcomfort.com/portal"

import re
import time, datetime
import httplib, urllib

class Thermostat(object):


	def __init__(self, USERNAME = 'zachary.swarth@gmail.com', PASSWORD = 'Zachary1', DEVICE_ID = 699427):
		self.USERNAME = USERNAME
		self.PASSWORD = PASSWORD
		self.DEVICE_ID = DEVICE_ID

	def client_cookies(self, cookiestr,container):
		cookiere=re.compile('\s*([^=]+)\s*=\s*([^;]*)\s*')
		if not container: container={}
		toks=re.split(';|,',cookiestr)
		for t in toks:
			k=None
			v=None
			m=cookiere.search(t)
			if m:
				k=m.group(1)
				v=m.group(2)
				if (k in ['path','Path','HttpOnly']):
					k=None
					v=None
			if k: 
				container[k]=v
		return container

	def export_cookiejar(self, jar):
		s=""
		for x in jar:
			s+='%s=%s;' % (x,jar[x])
		return s

	def get_login(self, action, value=None, hold_time=1):
		cookiejar=None

		headers={"Content-Type":"application/x-www-form-urlencoded", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
				"Accept-Encoding":"sdch",
				"Host":"mytotalconnectcomfort.com",
				"DNT":"1",
				"Origin":"https://mytotalconnectcomfort.com/portal",
				"User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/28.0.1500.95 Safari/537.36"
			}
		conn = httplib.HTTPSConnection("mytotalconnectcomfort.com")
		conn.request("GET", "/portal/",None,headers)
		r0 = conn.getresponse()
	    
		for x in r0.getheaders():
			(n,v) = x
	      #print "R0 HEADER",n,v
			if (n.lower() == "set-cookie"): 
				cookiejar=self.client_cookies(v,cookiejar)
	    #cookiejar = r0.getheader("Set-Cookie")
		location = r0.getheader("Location")

		retries=5
		params=urllib.urlencode({"timeOffset":"240",
	        "UserName":self.USERNAME,
	        "Password":self.PASSWORD,
	        "RememberMe":"false"})
	    #print params
		newcookie=self.export_cookiejar(cookiejar)
	    #print "Cookiejar now",newcookie
		headers={"Content-Type":"application/x-www-form-urlencoded",
	            "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
	            "Accept-Encoding":"sdch",
	            "Host":"mytotalconnectcomfort.com",
	            "DNT":"1",
	            "Origin":"https://mytotalconnectcomfort.com/portal/",
	            "Cookie":newcookie,
	            "User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/28.0.1500.95 Safari/537.36"
	        }
		conn = httplib.HTTPSConnection("mytotalconnectcomfort.com")
		conn.request("POST", "/portal/",params,headers)
		r1 = conn.getresponse()
	    #print r1.status, r1.reason
	    
		for x in r1.getheaders():
			(n,v) = x
	      #print "GOT2 HEADER",n,v
	    	if (n.lower() == "set-cookie"): 
				cookiejar=client_cookies(v,cookiejar)
		cookie=self.export_cookiejar(cookiejar)
		print "Cookiejar now",cookie
		location = r1.getheader("Location")

		if ((location == None) or (r1.status != 302)):
	        #raise BaseException("Login fail" )
			print("ErrorNever got redirect on initial login  status={0} {1}".format(r1.status,r1.reason))
			return

	   # Skip second query - just go directly to our device_id, rather than letting it
	    # redirect us to it. 

		code=str(self.DEVICE_ID)

		t = datetime.datetime.now()
		utc_seconds = (time.mktime(t.timetuple()))
		utc_seconds = int(utc_seconds*1000)
	    #print "Code ",code

		location="/portal/Device/CheckDataSession/"+code+"?_="+str(utc_seconds)
	    #print "THIRD"
		headers={
				"Accept":"*/*",
	            "DNT":"1",
	            #"Accept-Encoding":"gzip,deflate,sdch",
	            "Accept-Encoding":"plain",
	            "Cache-Control":"max-age=0",
	            "Accept-Language":"en-US,en,q=0.8",
	            "Connection":"keep-alive",
	            "Host":"mytotalconnectcomfort.com",
	            "Referer":"https://mytotalconnectcomfort.com/portal/",
	            "X-Requested-With":"XMLHttpRequest",
	            "User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/28.0.1500.95 Safari/537.36",
	            "Cookie":cookie
	        }
		conn = httplib.HTTPSConnection("mytotalconnectcomfort.com")
	    #conn.set_debuglevel(999);
	    #print "LOCATION R3 is",location
		conn.request("GET", location,None,headers)
		r3 = conn.getresponse()
		if (r3.status != 200):
			print("Error Didn't get 200 status on R3 status={0} {1}".format(r3.status,r3.reason))
			return


	    # Print thermostat information returned

		if (action == "status"):
	    
	        #print r3.status, r3.reason
			rawdata=r3.read()
			j = json.loads(rawdata)
	        #print "R3 Dump"
	        #print json.dumps(j,indent=2)
	        #print json.dumps(j,sort_keys=True,indent=4, separators=(',', ': '))
	        #print "Success:",j['success']
	        #print "Live",j['deviceLive']
			print "Indoor Temperature:",j['latestData']['uiData']["DispTemperature"]
			print "Indoor Humidity:",j['latestData']['uiData']["IndoorHumidity"]
			print "Cool Setpoint:",j['latestData']['uiData']["CoolSetpoint"]
			print "Heat Setpoint:",j['latestData']['uiData']["HeatSetpoint"]
			print "Hold Until :",j['latestData']['uiData']["TemporaryHoldUntilTime"]
			print "Status Cool:",j['latestData']['uiData']["StatusCool"]
			print "Status Heat:",j['latestData']['uiData']["StatusHeat"]
			print "Status Fan:",j['latestData']['fanData']["fanMode"]

			return
	    
		headers={
	            "Accept":'application/json; q=0.01',
	            "DNT":"1",
	            "Accept-Encoding":"gzip,deflate,sdch",
	            'Content-Type':'application/json; charset=UTF-8',
	            "Cache-Control":"max-age=0",
	            "Accept-Language":"en-US,en,q=0.8",
	            "Connection":"keep-alive",
	            "Host":"mytotalconnectcomfort.com",
	            "Referer":"https://mytotalconnectcomfort.com/portal/",
	            "X-Requested-With":"XMLHttpRequest",
	            "User-Agent":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/28.0.1500.95 Safari/537.36",
	            'Referer':"/TotalConnectComfort/Device/CheckDataSession/"+code,
	            "Cookie":cookie
	        }

	# get_login('status', value=None, hold_time=1)