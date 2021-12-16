# Module 10 Application

## Challenge: Crypto Clustering

In this Challenge, you’ll combine your financial Python programming skills with the new unsupervised learning skills that you acquired in this module.

The CSV file provided for this challenge contains price change data of cryptocurrencies in different periods.

The steps for this challenge are broken out into the following sections:

* Import the Data (provided in the starter code)
* Prepare the Data (provided in the starter code)
* Find the Best Value for `k` Using the Original Data
* Cluster Cryptocurrencies with K-means Using the Original Data
* Optimize Clusters with Principal Component Analysis
* Find the Best Value for `k` Using the PCA Data
* Cluster the Cryptocurrencies with K-means Using the PCA Data
* Visualize and Compare the Results

### Import the Data

This section imports the data into a new DataFrame. It follows these steps:

1. Read  the “crypto_market_data.csv” file from the Resources folder into a DataFrame, and use `index_col="coin_id"` to set the cryptocurrency name as the index. Review the DataFrame.

2. Generate the summary statistics, and use HvPlot to visualize your data to observe what your DataFrame contains.


> **Rewind:** The [Pandas`describe()`function](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) generates summary statistics for a DataFrame. 


```python
# Import required libraries and dependencies
import pandas as pd
import hvplot.pandas
from path import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```










```python
# Load the data into a Pandas DataFrame
df_market_data = pd.read_csv(
    Path("Resources/crypto_market_data.csv"),
    index_col="coin_id")

# Display sample data
df_market_data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_change_percentage_24h</th>
      <th>price_change_percentage_7d</th>
      <th>price_change_percentage_14d</th>
      <th>price_change_percentage_30d</th>
      <th>price_change_percentage_60d</th>
      <th>price_change_percentage_200d</th>
      <th>price_change_percentage_1y</th>
    </tr>
    <tr>
      <th>coin_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bitcoin</th>
      <td>1.08388</td>
      <td>7.60278</td>
      <td>6.57509</td>
      <td>7.67258</td>
      <td>-3.25185</td>
      <td>83.51840</td>
      <td>37.51761</td>
    </tr>
    <tr>
      <th>ethereum</th>
      <td>0.22392</td>
      <td>10.38134</td>
      <td>4.80849</td>
      <td>0.13169</td>
      <td>-12.88890</td>
      <td>186.77418</td>
      <td>101.96023</td>
    </tr>
    <tr>
      <th>tether</th>
      <td>-0.21173</td>
      <td>0.04935</td>
      <td>0.00640</td>
      <td>-0.04237</td>
      <td>0.28037</td>
      <td>-0.00542</td>
      <td>0.01954</td>
    </tr>
    <tr>
      <th>ripple</th>
      <td>-0.37819</td>
      <td>-0.60926</td>
      <td>2.24984</td>
      <td>0.23455</td>
      <td>-17.55245</td>
      <td>39.53888</td>
      <td>-16.60193</td>
    </tr>
    <tr>
      <th>bitcoin-cash</th>
      <td>2.90585</td>
      <td>17.09717</td>
      <td>14.75334</td>
      <td>15.74903</td>
      <td>-13.71793</td>
      <td>21.66042</td>
      <td>14.49384</td>
    </tr>
    <tr>
      <th>binancecoin</th>
      <td>2.10423</td>
      <td>12.85511</td>
      <td>6.80688</td>
      <td>0.05865</td>
      <td>36.33486</td>
      <td>155.61937</td>
      <td>69.69195</td>
    </tr>
    <tr>
      <th>chainlink</th>
      <td>-0.23935</td>
      <td>20.69459</td>
      <td>9.30098</td>
      <td>-11.21747</td>
      <td>-43.69522</td>
      <td>403.22917</td>
      <td>325.13186</td>
    </tr>
    <tr>
      <th>cardano</th>
      <td>0.00322</td>
      <td>13.99302</td>
      <td>5.55476</td>
      <td>10.10553</td>
      <td>-22.84776</td>
      <td>264.51418</td>
      <td>156.09756</td>
    </tr>
    <tr>
      <th>litecoin</th>
      <td>-0.06341</td>
      <td>6.60221</td>
      <td>7.28931</td>
      <td>1.21662</td>
      <td>-17.23960</td>
      <td>27.49919</td>
      <td>-12.66408</td>
    </tr>
    <tr>
      <th>bitcoin-cash-sv</th>
      <td>0.92530</td>
      <td>3.29641</td>
      <td>-1.86656</td>
      <td>2.88926</td>
      <td>-24.87434</td>
      <td>7.42562</td>
      <td>93.73082</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Generate summary statistics
df_market_data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_change_percentage_24h</th>
      <th>price_change_percentage_7d</th>
      <th>price_change_percentage_14d</th>
      <th>price_change_percentage_30d</th>
      <th>price_change_percentage_60d</th>
      <th>price_change_percentage_200d</th>
      <th>price_change_percentage_1y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>41.000000</td>
      <td>41.000000</td>
      <td>41.000000</td>
      <td>41.000000</td>
      <td>41.000000</td>
      <td>41.000000</td>
      <td>41.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.269686</td>
      <td>4.497147</td>
      <td>0.185787</td>
      <td>1.545693</td>
      <td>-0.094119</td>
      <td>236.537432</td>
      <td>347.667956</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.694793</td>
      <td>6.375218</td>
      <td>8.376939</td>
      <td>26.344218</td>
      <td>47.365803</td>
      <td>435.225304</td>
      <td>1247.842884</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-13.527860</td>
      <td>-6.094560</td>
      <td>-18.158900</td>
      <td>-34.705480</td>
      <td>-44.822480</td>
      <td>-0.392100</td>
      <td>-17.567530</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.608970</td>
      <td>0.047260</td>
      <td>-5.026620</td>
      <td>-10.438470</td>
      <td>-25.907990</td>
      <td>21.660420</td>
      <td>0.406170</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-0.063410</td>
      <td>3.296410</td>
      <td>0.109740</td>
      <td>-0.042370</td>
      <td>-7.544550</td>
      <td>83.905200</td>
      <td>69.691950</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.612090</td>
      <td>7.602780</td>
      <td>5.510740</td>
      <td>4.578130</td>
      <td>0.657260</td>
      <td>216.177610</td>
      <td>168.372510</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.840330</td>
      <td>20.694590</td>
      <td>24.239190</td>
      <td>140.795700</td>
      <td>223.064370</td>
      <td>2227.927820</td>
      <td>7852.089700</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot your data to see what's in your DataFrame
df_market_data.hvplot.line(
    width=800,
    height=400,
    rot=45
)
```






<div id='1002'>





  <div class="bk-root" id="aa9dc3b8-f770-414a-a1e5-d7ee6dc65de6" data-root-id="1002"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"b325a67c-79dd-472d-b9b2-a73fb3484665":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{"coordinates":null,"group":null,"text_color":"black","text_font_size":"12pt"},"id":"1014","type":"Title"},{"attributes":{"coordinates":null,"data_source":{"id":"1069"},"glyph":{"id":"1072"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1074"},"nonselection_glyph":{"id":"1073"},"selection_glyph":{"id":"1090"},"view":{"id":"1076"}},"id":"1075","type":"GlyphRenderer"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer01774","sizing_mode":"stretch_width"},"id":"1429","type":"Spacer"},{"attributes":{},"id":"1018","type":"CategoricalScale"},{"attributes":{},"id":"1110","type":"UnionRenderers"},{"attributes":{},"id":"1142","type":"Selection"},{"attributes":{"line_alpha":0.2,"line_color":"#fc4f30","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1074","type":"Line"},{"attributes":{"line_alpha":0.2,"line_color":"#6d904f","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1120","type":"Line"},{"attributes":{},"id":"1023","type":"CategoricalTicker"},{"attributes":{"source":{"id":"1069"}},"id":"1076","type":"CDSView"},{"attributes":{"axis":{"id":"1022"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"1024","type":"Grid"},{"attributes":{"line_color":"#6d904f","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1140","type":"Line"},{"attributes":{},"id":"1070","type":"Selection"},{"attributes":{},"id":"1020","type":"LinearScale"},{"attributes":{},"id":"1086","type":"UnionRenderers"},{"attributes":{},"id":"1044","type":"AllLabels"},{"attributes":{"line_alpha":0.1,"line_color":"#fc4f30","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1073","type":"Line"},{"attributes":{},"id":"1170","type":"Selection"},{"attributes":{"axis":{"id":"1025"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"1028","type":"Grid"},{"attributes":{"axis_label":"coin_id","coordinates":null,"formatter":{"id":"1043"},"group":null,"major_label_orientation":0.7853981633974483,"major_label_policy":{"id":"1044"},"ticker":{"id":"1023"}},"id":"1022","type":"CategoricalAxis"},{"attributes":{},"id":"1030","type":"PanTool"},{"attributes":{"data":{"Variable":["price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h","price_change_percentage_24h"],"coin_id":["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"],"value":{"__ndarray__":"nZ0MjpJX8T8MzXUaaanMP5mByvj3Gcu/wCZr1EM02L8ep+hILj8HQL0Yyol21QBAwhcmUwWjzr//CS5W1GBqP7JGPUSjO7C/QKTfvg6c7T8urYbEPZbjPzeJQWDl0Ma/1pC4x9KHwj87NgLxun7bP90HILWJk7M/JjYf14aK5b83iUFg5dAFwG3i5H6HIvC/5nlwd9Zu7r9i83FtqBjLP+iHEcKjjd8/M9yAzw+j8T/fGtgqweLAv+RmuAGfH9q/UdobfGEy678D7KNTVz67v8YzaOif4No/teBFX0Ga9D8G2Eenrnzjvx+duvJZPhLAoMN8eQH28L+cxCCwcmjdv667eapDDivAX5hMFYxK479EUaBP5EkQwAZkr3d/XBNA0JuKVBgbBEDqBDQRNrz1vxo09E9wseo/tI6qJoi6r79qMA3DR8QHQA==","dtype":"float64","order":"little","shape":[41]}},"selected":{"id":"1049"},"selection_policy":{"id":"1063"}},"id":"1048","type":"ColumnDataSource"},{"attributes":{"label":{"value":"price_change_percentage_7d"},"renderers":[{"id":"1075"}]},"id":"1089","type":"LegendItem"},{"attributes":{},"id":"1116","type":"Selection"},{"attributes":{"axis_label":"","coordinates":null,"formatter":{"id":"1046"},"group":null,"major_label_policy":{"id":"1047"},"ticker":{"id":"1026"}},"id":"1025","type":"LinearAxis"},{"attributes":{"line_alpha":0.2,"line_color":"#8b8b8b","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1146","type":"Line"},{"attributes":{},"id":"1026","type":"BasicTicker"},{"attributes":{"line_color":"#fc4f30","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1072","type":"Line"},{"attributes":{},"id":"1031","type":"WheelZoomTool"},{"attributes":{"source":{"id":"1141"}},"id":"1148","type":"CDSView"},{"attributes":{"coordinates":null,"data_source":{"id":"1141"},"glyph":{"id":"1144"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1146"},"nonselection_glyph":{"id":"1145"},"selection_glyph":{"id":"1168"},"view":{"id":"1148"}},"id":"1147","type":"GlyphRenderer"},{"attributes":{},"id":"1029","type":"SaveTool"},{"attributes":{},"id":"1200","type":"Selection"},{"attributes":{"line_alpha":0.1,"line_color":"#8b8b8b","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1145","type":"Line"},{"attributes":{"line_color":"#17becf","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1198","type":"Line"},{"attributes":{"overlay":{"id":"1034"}},"id":"1032","type":"BoxZoomTool"},{"attributes":{"coordinates":null,"data_source":{"id":"1199"},"glyph":{"id":"1202"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1204"},"nonselection_glyph":{"id":"1203"},"selection_glyph":{"id":"1230"},"view":{"id":"1206"}},"id":"1205","type":"GlyphRenderer"},{"attributes":{},"id":"1033","type":"ResetTool"},{"attributes":{},"id":"1164","type":"UnionRenderers"},{"attributes":{"label":{"value":"price_change_percentage_60d"},"renderers":[{"id":"1147"}]},"id":"1167","type":"LegendItem"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"1034","type":"BoxAnnotation"},{"attributes":{"line_alpha":0.1,"line_color":"#17becf","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1173","type":"Line"},{"attributes":{"line_color":"#8b8b8b","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1144","type":"Line"},{"attributes":{},"id":"1063","type":"UnionRenderers"},{"attributes":{"coordinates":null,"data_source":{"id":"1169"},"glyph":{"id":"1172"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1174"},"nonselection_glyph":{"id":"1173"},"selection_glyph":{"id":"1198"},"view":{"id":"1176"}},"id":"1175","type":"GlyphRenderer"},{"attributes":{},"id":"1092","type":"Selection"},{"attributes":{"source":{"id":"1199"}},"id":"1206","type":"CDSView"},{"attributes":{"data":{"Variable":["price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d","price_change_percentage_30d"],"coin_id":["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"],"value":{"__ndarray__":"VMa/z7iwHkAl6ZrJN9vAP/28qUiFsaW/L26jAbwFzj8dcjPcgH8vQPMf0m9fB64/5bhTOlhvJsBnfjUHCDYkQGWNeohGd/M/sCDNWDQdB0Cmft5UpHIxwJgvL8A+OsW/t0WZDTKJEsDeVKTC2HpDQEzD8BExNSnAjWK5pdVALMDjjcwjf+AkwIrIsIo3QiDAjliLTwFQEkAWpBmLpnMBwE3WqIdo9DXAVG8NbJWAHUAnMQisHJoIQGA8g4b+CQrACcTr+gU7DcCPpQ9dUN+SP80Bgjl6nBDA9S1zuixWLMBC7Eyh83odQEku/yH9bj9AtTf4wmQqH8Am/FI/byoHQNjYJaq3/j3A9GxWfa62MECdRloqb9c0wF4R/G8lOyLArK3YX3aZYUC1/SsrTVpBwP/PYb68ICXAFR3J5T+knz+ndLD+z4EqQA==","dtype":"float64","order":"little","shape":[41]}},"selected":{"id":"1116"},"selection_policy":{"id":"1136"}},"id":"1115","type":"ColumnDataSource"},{"attributes":{"line_alpha":0.2,"line_color":"#9467bd","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1204","type":"Line"},{"attributes":{"coordinates":null,"data_source":{"id":"1091"},"glyph":{"id":"1094"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1096"},"nonselection_glyph":{"id":"1095"},"selection_glyph":{"id":"1114"},"view":{"id":"1098"}},"id":"1097","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.1,"line_color":"#9467bd","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1203","type":"Line"},{"attributes":{},"id":"1226","type":"UnionRenderers"},{"attributes":{"line_color":"#9467bd","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1230","type":"Line"},{"attributes":{"source":{"id":"1091"}},"id":"1098","type":"CDSView"},{"attributes":{"label":{"value":"price_change_percentage_1y"},"renderers":[{"id":"1205"}]},"id":"1229","type":"LegendItem"},{"attributes":{"line_color":"#9467bd","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1202","type":"Line"},{"attributes":{"line_color":"#e5ae38","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1114","type":"Line"},{"attributes":{},"id":"1043","type":"CategoricalTickFormatter"},{"attributes":{"line_color":"#fc4f30","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1090","type":"Line"},{"attributes":{},"id":"1047","type":"AllLabels"},{"attributes":{"tools":[{"id":"1006"},{"id":"1029"},{"id":"1030"},{"id":"1031"},{"id":"1032"},{"id":"1033"}]},"id":"1035","type":"Toolbar"},{"attributes":{"factors":["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"],"tags":[[["coin_id","coin_id",null]]]},"id":"1004","type":"FactorRange"},{"attributes":{"line_color":"#8b8b8b","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1168","type":"Line"},{"attributes":{"coordinates":null,"data_source":{"id":"1115"},"glyph":{"id":"1118"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1120"},"nonselection_glyph":{"id":"1119"},"selection_glyph":{"id":"1140"},"view":{"id":"1122"}},"id":"1121","type":"GlyphRenderer"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer01773","sizing_mode":"stretch_width"},"id":"1003","type":"Spacer"},{"attributes":{"label":{"value":"price_change_percentage_24h"},"renderers":[{"id":"1054"}]},"id":"1067","type":"LegendItem"},{"attributes":{"line_color":"#e5ae38","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1094","type":"Line"},{"attributes":{"label":{"value":"price_change_percentage_30d"},"renderers":[{"id":"1121"}]},"id":"1139","type":"LegendItem"},{"attributes":{"end":8641.780918,"reset_end":8641.780918,"reset_start":-834.5136980000001,"start":-834.5136980000001,"tags":[[["value","value",null]]]},"id":"1005","type":"Range1d"},{"attributes":{"children":[{"id":"1003"},{"id":"1013"},{"id":"1429"}],"margin":[0,0,0,0],"name":"Row01769","tags":["embedded"]},"id":"1002","type":"Row"},{"attributes":{"line_alpha":0.2,"line_color":"#30a2da","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1053","type":"Line"},{"attributes":{"data":{"Variable":["price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d","price_change_percentage_60d"],"coin_id":["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"],"value":{"__ndarray__":"FvvL7skDCsAs1JrmHccpwKjjMQOV8dE/2qz6XG2NMcBzol2FlG8rwIleRrHcKkJAN8MN+PzYRcBi26LMBtk2wCEf9GxWPTHAq5UJv9TfOMBg5dAi23kwwDygbMoV3qU/l631RUIrPsDNzMzMzNxEQJm7lpAPGhZAms5OBkdpRsBKe4MvTKYGwEsfuqC+zT7APL1SliHOHMC4AZ8fRgjlP9L7xteeCStAFqQZi6azDMCvJeSDni0ewDeOWItPQQvAXvQVpBm3VEDaOGItPgW4PwpLPKBsQkHAaw4QzNHPRcAJM23/yoo0wBmQvd79AVRAfa62Yn85OsBVGFsIcug5wLCsNCkFHQFA529CIQKeMsA0uoPYmbZDwPKwUGua0VNA/pqsUQ/ia0Ao8iTpmllAwFuxv+yenBdAmrFoOjsZ0D8s1JrmHRc/wA==","dtype":"float64","order":"little","shape":[41]}},"selected":{"id":"1142"},"selection_policy":{"id":"1164"}},"id":"1141","type":"ColumnDataSource"},{"attributes":{},"id":"1046","type":"BasicTickFormatter"},{"attributes":{},"id":"1136","type":"UnionRenderers"},{"attributes":{"line_alpha":0.2,"line_color":"#17becf","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1174","type":"Line"},{"attributes":{"source":{"id":"1169"}},"id":"1176","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_color":"#6d904f","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1119","type":"Line"},{"attributes":{"callback":null,"renderers":[{"id":"1054"},{"id":"1075"},{"id":"1097"},{"id":"1121"},{"id":"1147"},{"id":"1175"},{"id":"1205"}],"tags":["hv_created"],"tooltips":[["Variable","@{Variable}"],["coin_id","@{coin_id}"],["value","@{value}"]]},"id":"1006","type":"HoverTool"},{"attributes":{},"id":"1049","type":"Selection"},{"attributes":{},"id":"1194","type":"UnionRenderers"},{"attributes":{"line_color":"#6d904f","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1118","type":"Line"},{"attributes":{"below":[{"id":"1022"}],"center":[{"id":"1024"},{"id":"1028"}],"height":400,"left":[{"id":"1025"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"1054"},{"id":"1075"},{"id":"1097"},{"id":"1121"},{"id":"1147"},{"id":"1175"},{"id":"1205"}],"right":[{"id":"1066"}],"sizing_mode":"fixed","title":{"id":"1014"},"toolbar":{"id":"1035"},"width":800,"x_range":{"id":"1004"},"x_scale":{"id":"1018"},"y_range":{"id":"1005"},"y_scale":{"id":"1020"}},"id":"1013","subtype":"Figure","type":"Plot"},{"attributes":{"line_color":"#17becf","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1172","type":"Line"},{"attributes":{"source":{"id":"1115"}},"id":"1122","type":"CDSView"},{"attributes":{"label":{"value":"price_change_percentage_200d"},"renderers":[{"id":"1175"}]},"id":"1197","type":"LegendItem"},{"attributes":{"line_alpha":0.1,"line_color":"#e5ae38","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1095","type":"Line"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1051","type":"Line"},{"attributes":{"data":{"Variable":["price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y","price_change_percentage_1y"],"coin_id":["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"],"value":{"__ndarray__":"kQpjC0HCQkDWqIdodH1ZQN/42jNLApQ/NXugFRiaMMDN6bKY2PwsQOcdp+hIbFFAaTo7GRxSdEC6LCY2H4NjQPnaM0sCVCnA0NA/wcVuV0CQvd798RBzQL4wmSoYlci/t39lpUmRMcDovMYuUaFhQEPKT6p9nk1A6PaSxmiAYUBdUN8yp75hQMzuycNCnStAw7ZFmQ2cVEAHsTOFzmvSvx/0bFZ9aWBAhhvw+WHEQkAN/RNcrAA1QN8Vwf9WkjhAS7A4nPkWaUAVUn5S7dPBPzUk7rH04Q/AIsMq3shbaUBf0hito7hUQPyMCwcC64VAzsKedvirA8Dwoq8gzTgmQHRGlPYWrL5Ayk+qfTpOJUDfiVkvRjaFQEfJq3MMC3RAnFCIgONmn0CsVib8Uk8pwNttF5rrC2VAyXGndLD+2T/ZfFwbKgxnQA==","dtype":"float64","order":"little","shape":[41]}},"selected":{"id":"1200"},"selection_policy":{"id":"1226"}},"id":"1199","type":"ColumnDataSource"},{"attributes":{"label":{"value":"price_change_percentage_14d"},"renderers":[{"id":"1097"}]},"id":"1113","type":"LegendItem"},{"attributes":{"coordinates":null,"data_source":{"id":"1048"},"glyph":{"id":"1051"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1053"},"nonselection_glyph":{"id":"1052"},"selection_glyph":{"id":"1068"},"view":{"id":"1055"}},"id":"1054","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.2,"line_color":"#e5ae38","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1096","type":"Line"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1068","type":"Line"},{"attributes":{"data":{"Variable":["price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d","price_change_percentage_14d"],"coin_id":["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"],"value":{"__ndarray__":"zQaZZORMGkB5knTN5DsTQC1DHOviNno/ctwpHaz/AUCLic3HtYEtQJgvL8A+OhtACoDxDBqaIkAOhGQBEzgWQICfceFAKB1AIVnABG7d/b8xsfm4NhQhwGtI3GPpQ3e/pfeNrz0z8j8ao3VUNWkyQBUA4xk0dPY/VIzzN6GQAkCXrfVFQhsUwKpla32RUPG/Qgkzbf8KFkDPg7uzdtvFP8MN+PwwsiPAYFlpUgo6GkB3+GuyRh0cwM2v5gDBHN0/jliLTwEwBUDWrZ6T3jeuP7pOIy2VNw7Am1Wfq60oMsAJM23/yioXQJfK2xFOSxrAQE0tW+uL4D9OucK7XET0v0SLbOf7mSPA/pqsUQ/R978vaYzWUZUqwDtT6LzGbirAvodLjjs9OEByUMJM2x8lwJ30vvG1Z8q/ZCMQr+sXvD+7D0BqE0cdwA==","dtype":"float64","order":"little","shape":[41]}},"selected":{"id":"1092"},"selection_policy":{"id":"1110"}},"id":"1091","type":"ColumnDataSource"},{"attributes":{"data":{"Variable":["price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d","price_change_percentage_200d"],"coin_id":["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"],"value":{"__ndarray__":"xY8xdy3hVED0piIVxlhnQBe86CtIM3a/9zsUBfrEQ0DTn/1IEak1QPTDCOHRc2NAl3MprqozeUAexM4UOohwQIB9dOrKfztAFhiyutWzHUCN7iB2plZsQIQqNXugFcS/ZHWr56QPMkALe9rhrzBlQH+8V61MnGBA5nlwd9a2RUARHm0csfpDQC+Lic3HJ1VAkj8YeO4/Z0AOvjCZKhjZv2vUQzS61GNAtRX7y+75VEC5GW7A52cwQAPso1NXHkVAKa4q+64Fa0A7NgLxun65P+TaUDHOwVdAyM1wAz7bcEAFwHgGDaNeQH3Qs1k1lYtAOUVHcvnvGEANGvonuMgkQCYZOQvbZ6FAGD4ipkSiVED5MeauZXeDQHi0ccQaSoJAC170FcTYmEB1PGagMmJLQC2yne+nvkxAndfYJaq3tr9R9wFIbSl+QA==","dtype":"float64","order":"little","shape":[41]}},"selected":{"id":"1170"},"selection_policy":{"id":"1194"}},"id":"1169","type":"ColumnDataSource"},{"attributes":{"data":{"Variable":["price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d","price_change_percentage_7d"],"coin_id":["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"],"value":{"__ndarray__":"RbsKKT9pHkAzUBn/PsMkQL99HThnRKk/yVnY0w5/47/+JhQi4BgxQBL3WPrQtSlA7yB2ptCxNEA/jBAebfwrQPFL/bypaBpArKjBNAxfCkByv0NRoK8WwPlJtU/HY76/lPsdigL99L+n6Egu/5EvQE1KQbeX9BBA0m9fB87pIUBlU67wLjcSwKbtX1lpkgBAZwqd19gVMEBcIEHxY8ytP3wnZr0YigNAOh4zUBmfHUD3Hi457pT1vzNQGf8+4/k/kQ96Nqs+6781Y9F0djKoP2dEaW/wRRhAgXhdv2A3/L/8GHPXErIkQP8JLlbUYBjAVU0QdR9gFEDRlnMprmoIQGCrBIvD2RBAAiuHFtlOIECbG9MTllgRQKhXyjLEURtAqn06HjNQ4z+UvDrHgGzzvwltOZfiahxAsD2zJEBNxT8yj/zBwHPlPw==","dtype":"float64","order":"little","shape":[41]}},"selected":{"id":"1070"},"selection_policy":{"id":"1086"}},"id":"1069","type":"ColumnDataSource"},{"attributes":{"source":{"id":"1048"}},"id":"1055","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_color":"#30a2da","line_width":2,"x":{"field":"coin_id"},"y":{"field":"value"}},"id":"1052","type":"Line"},{"attributes":{"click_policy":"mute","coordinates":null,"group":null,"items":[{"id":"1067"},{"id":"1089"},{"id":"1113"},{"id":"1139"},{"id":"1167"},{"id":"1197"},{"id":"1229"}],"location":[0,0],"title":"Variable"},"id":"1066","type":"Legend"}],"root_ids":["1002"]},"title":"Bokeh Application","version":"2.4.1"}};
    var render_items = [{"docid":"b325a67c-79dd-472d-b9b2-a73fb3484665","root_ids":["1002"],"roots":{"1002":"aa9dc3b8-f770-414a-a1e5-d7ee6dc65de6"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>



---

### Prepare the Data

This section prepares the data before running the K-Means algorithm. It follows these steps:

1. Use the `StandardScaler` module from scikit-learn to normalize the CSV file data. This will require you to utilize the `fit_transform` function.

2. Create a DataFrame that contains the scaled data. Be sure to set the `coin_id` index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.



```python
# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
scaled_data = StandardScaler().fit_transform(df_market_data)
```


```python
# Create a DataFrame with the scaled data
df_market_data_scaled = pd.DataFrame(
    scaled_data,
    columns=df_market_data.columns
)

# Copy the crypto names from the original data
df_market_data_scaled["coin_id"] = df_market_data.index

# Set the coinid column as index
df_market_data_scaled = df_market_data_scaled.set_index("coin_id")

# Display sample data
df_market_data_scaled.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_change_percentage_24h</th>
      <th>price_change_percentage_7d</th>
      <th>price_change_percentage_14d</th>
      <th>price_change_percentage_30d</th>
      <th>price_change_percentage_60d</th>
      <th>price_change_percentage_200d</th>
      <th>price_change_percentage_1y</th>
    </tr>
    <tr>
      <th>coin_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bitcoin</th>
      <td>0.508529</td>
      <td>0.493193</td>
      <td>0.772200</td>
      <td>0.235460</td>
      <td>-0.067495</td>
      <td>-0.355953</td>
      <td>-0.251637</td>
    </tr>
    <tr>
      <th>ethereum</th>
      <td>0.185446</td>
      <td>0.934445</td>
      <td>0.558692</td>
      <td>-0.054341</td>
      <td>-0.273483</td>
      <td>-0.115759</td>
      <td>-0.199352</td>
    </tr>
    <tr>
      <th>tether</th>
      <td>0.021774</td>
      <td>-0.706337</td>
      <td>-0.021680</td>
      <td>-0.061030</td>
      <td>0.008005</td>
      <td>-0.550247</td>
      <td>-0.282061</td>
    </tr>
    <tr>
      <th>ripple</th>
      <td>-0.040764</td>
      <td>-0.810928</td>
      <td>0.249458</td>
      <td>-0.050388</td>
      <td>-0.373164</td>
      <td>-0.458259</td>
      <td>-0.295546</td>
    </tr>
    <tr>
      <th>bitcoin-cash</th>
      <td>1.193036</td>
      <td>2.000959</td>
      <td>1.760610</td>
      <td>0.545842</td>
      <td>-0.291203</td>
      <td>-0.499848</td>
      <td>-0.270317</td>
    </tr>
  </tbody>
</table>
</div>



---

### Find the Best Value for k Using the Original Data

In this section, you will use the elbow method to find the best value for `k`.

1. Code the elbow method algorithm to find the best value for `k`. Use a range from 1 to 11. 

2. Plot a line chart with all the inertia values computed with the different values of `k` to visually identify the optimal value for `k`.

3. Answer the following question: What is the best value for `k`?


```python
# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))
```


```python
# Create an empy list to store the inertia values
inertia = []
```


```python
# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_scaled`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    model = KMeans(n_clusters=i, random_state=0)
    model.fit(df_market_data_scaled)
    inertia.append(model.inertia_)
    
inertia
```

    C:\Users\ijisk\anaconda3\lib\site-packages\sklearn\cluster\_kmeans.py:881: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    




    [287.0,
     195.82021818036043,
     123.19048183836959,
     79.02243535120978,
     65.30237914162501,
     52.88851821977533,
     43.91469044757748,
     37.51703249010357,
     32.48524083218354,
     28.222899290608932]




```python
# Create a dictionary with the data to plot the Elbow curve
elbow_data_1 = { 
    "k": k,
    "inertia": inertia
}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow_data_1 = pd.DataFrame(elbow_data_1)
```


```python
df_elbow_data_1.dtypes
```




    k            int64
    inertia    float64
    dtype: object




```python
df_crypto_pca.dtypes
```




    PC1    float64
    PC2    float64
    PC3    float64
    dtype: object




```python
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
df_elbow_data_1.hvplot.line(
    x="k",
    y="inertia",
    title="Elbow Curve",
    xticks=k
)
```






<div id='1541'>





  <div class="bk-root" id="d71725a3-41d7-4e30-aec7-a52b81506e91" data-root-id="1541"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"20106bce-81f4-4c81-93fd-b7071365e78c":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{},"id":"1588","type":"AllLabels"},{"attributes":{},"id":"1587","type":"BasicTickFormatter"},{"attributes":{},"id":"1551","type":"LinearScale"},{"attributes":{"ticks":[1,2,3,4,5,6,7,8,9,10]},"id":"1585","type":"FixedTicker"},{"attributes":{"below":[{"id":"1555"}],"center":[{"id":"1558"},{"id":"1562"}],"height":300,"left":[{"id":"1559"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"1582"}],"sizing_mode":"fixed","title":{"id":"1547"},"toolbar":{"id":"1569"},"width":700,"x_range":{"id":"1543"},"x_scale":{"id":"1551"},"y_range":{"id":"1544"},"y_scale":{"id":"1553"}},"id":"1546","subtype":"Figure","type":"Plot"},{"attributes":{"end":312.8777100709391,"reset_end":312.8777100709391,"reset_start":2.3451892196698267,"start":2.3451892196698267,"tags":[[["inertia","inertia",null]]]},"id":"1544","type":"Range1d"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02889","sizing_mode":"stretch_width"},"id":"1542","type":"Spacer"},{"attributes":{"line_alpha":0.1,"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"1580","type":"Line"},{"attributes":{},"id":"1600","type":"UnionRenderers"},{"attributes":{"end":10.0,"reset_end":10.0,"reset_start":1.0,"start":1.0,"tags":[[["k","k",null]]]},"id":"1543","type":"Range1d"},{"attributes":{},"id":"1553","type":"LinearScale"},{"attributes":{},"id":"1592","type":"BasicTickFormatter"},{"attributes":{"coordinates":null,"group":null,"text":"Elbow Curve","text_color":"black","text_font_size":"12pt"},"id":"1547","type":"Title"},{"attributes":{"children":[{"id":"1542"},{"id":"1546"},{"id":"1612"}],"margin":[0,0,0,0],"name":"Row02885","tags":["embedded"]},"id":"1541","type":"Row"},{"attributes":{"coordinates":null,"data_source":{"id":"1576"},"glyph":{"id":"1579"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1581"},"nonselection_glyph":{"id":"1580"},"selection_glyph":{"id":"1584"},"view":{"id":"1583"}},"id":"1582","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"1555"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"1558","type":"Grid"},{"attributes":{},"id":"1577","type":"Selection"},{"attributes":{"data":{"inertia":{"__ndarray__":"AAAAAADwcUByhzI6P3poQOGRvNowzF5AGe6ulG/BU0BoEQsuWlNQQOfqC/e6cUpAiSebkxT1RUBt++EeLsJCQMF0IF8cPkBAi3eL7Q85PEA=","dtype":"float64","order":"little","shape":[10]},"k":[1,2,3,4,5,6,7,8,9,10]},"selected":{"id":"1577"},"selection_policy":{"id":"1600"}},"id":"1576","type":"ColumnDataSource"},{"attributes":{"axis_label":"k","coordinates":null,"formatter":{"id":"1587"},"group":null,"major_label_policy":{"id":"1588"},"ticker":{"id":"1585"}},"id":"1555","type":"LinearAxis"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer02890","sizing_mode":"stretch_width"},"id":"1612","type":"Spacer"},{"attributes":{"axis":{"id":"1559"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"1562","type":"Grid"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"1579","type":"Line"},{"attributes":{},"id":"1564","type":"PanTool"},{"attributes":{"axis_label":"inertia","coordinates":null,"formatter":{"id":"1592"},"group":null,"major_label_policy":{"id":"1593"},"ticker":{"id":"1560"}},"id":"1559","type":"LinearAxis"},{"attributes":{},"id":"1560","type":"BasicTicker"},{"attributes":{},"id":"1565","type":"WheelZoomTool"},{"attributes":{},"id":"1563","type":"SaveTool"},{"attributes":{"overlay":{"id":"1568"}},"id":"1566","type":"BoxZoomTool"},{"attributes":{},"id":"1567","type":"ResetTool"},{"attributes":{"line_alpha":0.2,"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"1581","type":"Line"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"1568","type":"BoxAnnotation"},{"attributes":{"source":{"id":"1576"}},"id":"1583","type":"CDSView"},{"attributes":{"tools":[{"id":"1545"},{"id":"1563"},{"id":"1564"},{"id":"1565"},{"id":"1566"},{"id":"1567"}]},"id":"1569","type":"Toolbar"},{"attributes":{"callback":null,"renderers":[{"id":"1582"}],"tags":["hv_created"],"tooltips":[["k","@{k}"],["inertia","@{inertia}"]]},"id":"1545","type":"HoverTool"},{"attributes":{},"id":"1593","type":"AllLabels"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"1584","type":"Line"}],"root_ids":["1541"]},"title":"Bokeh Application","version":"2.4.1"}};
    var render_items = [{"docid":"20106bce-81f4-4c81-93fd-b7071365e78c","root_ids":["1541"],"roots":{"1541":"d71725a3-41d7-4e30-aec7-a52b81506e91"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>



#### Answer the following question: What is the best value for k?
**Question:** What is the best value for `k`?

**Answer:** The best value for K is 4.

---

### Cluster Cryptocurrencies with K-means Using the Original Data

In this section, you will use the K-Means algorithm with the best value for `k` found in the previous section to cluster the cryptocurrencies according to the price changes of cryptocurrencies provided.

1. Initialize the K-Means model with four clusters using the best value for `k`. 

2. Fit the K-Means model using the original data.

3. Predict the clusters to group the cryptocurrencies using the original data. View the resulting array of cluster values.

4. Create a copy of the original data and add a new column with the predicted clusters.

5. Create a scatter plot using hvPlot by setting `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. Color the graph points with the labels found using K-Means and add the crypto name in the `hover_cols` parameter to identify the cryptocurrency represented by each data point.


```python
# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4)
```


```python
# Fit the K-Means model using the scaled data
model.fit(df_market_data_scaled)
```




    KMeans(n_clusters=4)




```python
# Predict the clusters to group the cryptocurrencies using the scaled data
crypto_clusters = model.predict(df_market_data_scaled)

# View the resulting array of cluster values.
crypto_clusters
```




    array([3, 3, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 0, 3, 0, 3, 0, 0, 3, 0, 0, 3,
           0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 3, 0, 0, 2, 0, 0, 0, 0])




```python
# Create a copy of the DataFrame
df_crypto_scaled_predictions = df_market_data_scaled.copy()
```


```python
# Add a new column to the DataFrame with the predicted clusters
df_crypto_scaled_predictions["CryptoCluster"] = crypto_clusters

# Display sample data
df_crypto_scaled_predictions.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_change_percentage_24h</th>
      <th>price_change_percentage_7d</th>
      <th>price_change_percentage_14d</th>
      <th>price_change_percentage_30d</th>
      <th>price_change_percentage_60d</th>
      <th>price_change_percentage_200d</th>
      <th>price_change_percentage_1y</th>
      <th>CryptoCluster</th>
    </tr>
    <tr>
      <th>coin_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bitcoin</th>
      <td>0.508529</td>
      <td>0.493193</td>
      <td>0.772200</td>
      <td>0.235460</td>
      <td>-0.067495</td>
      <td>-0.355953</td>
      <td>-0.251637</td>
      <td>3</td>
    </tr>
    <tr>
      <th>ethereum</th>
      <td>0.185446</td>
      <td>0.934445</td>
      <td>0.558692</td>
      <td>-0.054341</td>
      <td>-0.273483</td>
      <td>-0.115759</td>
      <td>-0.199352</td>
      <td>3</td>
    </tr>
    <tr>
      <th>tether</th>
      <td>0.021774</td>
      <td>-0.706337</td>
      <td>-0.021680</td>
      <td>-0.061030</td>
      <td>0.008005</td>
      <td>-0.550247</td>
      <td>-0.282061</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ripple</th>
      <td>-0.040764</td>
      <td>-0.810928</td>
      <td>0.249458</td>
      <td>-0.050388</td>
      <td>-0.373164</td>
      <td>-0.458259</td>
      <td>-0.295546</td>
      <td>0</td>
    </tr>
    <tr>
      <th>bitcoin-cash</th>
      <td>1.193036</td>
      <td>2.000959</td>
      <td>1.760610</td>
      <td>0.545842</td>
      <td>-0.291203</td>
      <td>-0.499848</td>
      <td>-0.270317</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
df_crypto_scaled_predictions.hvplot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    by="CryptoCluster",
    hover_cols = ["coin_id", "price_change_percentage_24h", "price_change_percentage_7d"],
    title="Scatter Plot by Crypto Segment - k4"
)
```






<div id='1659'>





  <div class="bk-root" id="8d2d71c9-5896-478e-9bb7-93d9614facda" data-root-id="1659"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"e05261df-dd91-44b5-b7cd-f19891b15873":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{"source":{"id":"1703"}},"id":"1710","type":"CDSView"},{"attributes":{},"id":"1672","type":"LinearScale"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#6d904f"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#6d904f"},"line_alpha":{"value":0.2},"line_color":{"value":"#6d904f"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"1775","type":"Scatter"},{"attributes":{"callback":null,"renderers":[{"id":"1709"},{"id":"1730"},{"id":"1752"},{"id":"1776"}],"tags":["hv_created"],"tooltips":[["CryptoCluster","@{CryptoCluster}"],["price_change_percentage_24h","@{price_change_percentage_24h}"],["price_change_percentage_7d","@{price_change_percentage_7d}"],["coin_id","@{coin_id}"]]},"id":"1663","type":"HoverTool"},{"attributes":{"source":{"id":"1770"}},"id":"1777","type":"CDSView"},{"attributes":{"label":{"value":"0"},"renderers":[{"id":"1709"}]},"id":"1722","type":"LegendItem"},{"attributes":{"click_policy":"mute","coordinates":null,"group":null,"items":[{"id":"1722"},{"id":"1744"},{"id":"1768"},{"id":"1794"}],"location":[0,0],"title":"CryptoCluster"},"id":"1721","type":"Legend"},{"attributes":{},"id":"1791","type":"UnionRenderers"},{"attributes":{"coordinates":null,"data_source":{"id":"1770"},"glyph":{"id":"1773"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1775"},"nonselection_glyph":{"id":"1774"},"selection_glyph":{"id":"1795"},"view":{"id":"1777"}},"id":"1776","type":"GlyphRenderer"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#30a2da"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#30a2da"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"circle"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"1723","type":"Scatter"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#fc4f30"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#fc4f30"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"circle"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"1745","type":"Scatter"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#e5ae38"},"line_alpha":{"value":0.1},"line_color":{"value":"#e5ae38"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"1750","type":"Scatter"},{"attributes":{"label":{"value":"3"},"renderers":[{"id":"1776"}]},"id":"1794","type":"LegendItem"},{"attributes":{},"id":"1674","type":"LinearScale"},{"attributes":{},"id":"1771","type":"Selection"},{"attributes":{"below":[{"id":"1676"}],"center":[{"id":"1679"},{"id":"1683"}],"height":300,"left":[{"id":"1680"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"1709"},{"id":"1730"},{"id":"1752"},{"id":"1776"}],"right":[{"id":"1721"}],"sizing_mode":"fixed","title":{"id":"1668"},"toolbar":{"id":"1690"},"width":700,"x_range":{"id":"1661"},"x_scale":{"id":"1672"},"y_range":{"id":"1662"},"y_scale":{"id":"1674"}},"id":"1667","subtype":"Figure","type":"Plot"},{"attributes":{"coordinates":null,"group":null,"text":"Scatter Plot by Crypto Segment - k4","text_color":"black","text_font_size":"12pt"},"id":"1668","type":"Title"},{"attributes":{"coordinates":null,"data_source":{"id":"1746"},"glyph":{"id":"1749"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1751"},"nonselection_glyph":{"id":"1750"},"selection_glyph":{"id":"1769"},"view":{"id":"1753"}},"id":"1752","type":"GlyphRenderer"},{"attributes":{"data":{"CryptoCluster":[2],"coin_id":["celsius-degree-token"],"price_change_percentage_24h":{"__ndarray__":"VLqXBn668D8=","dtype":"float64","order":"little","shape":[1]},"price_change_percentage_7d":{"__ndarray__":"Nv03JFjJ478=","dtype":"float64","order":"little","shape":[1]}},"selected":{"id":"1747"},"selection_policy":{"id":"1765"}},"id":"1746","type":"ColumnDataSource"},{"attributes":{},"id":"1677","type":"BasicTicker"},{"attributes":{"axis":{"id":"1676"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"1679","type":"Grid"},{"attributes":{"fill_color":{"value":"#e5ae38"},"hatch_color":{"value":"#e5ae38"},"line_color":{"value":"#e5ae38"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"1749","type":"Scatter"},{"attributes":{"data":{"CryptoCluster":[3,3,3,3,3,3,3,3,3,3,3,3,3],"coin_id":["bitcoin","ethereum","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","monero","tezos","cosmos","wrapped-bitcoin","zcash","maker"],"price_change_percentage_24h":{"__ndarray__":"DdlqYN9F4D86azDosLzHPzALIwCtFvM/9w6McDSK7D/VDHOpb1eHPzMEvaFkP7o/Sds39trWsz9Epj8/c9DQP4xAeNEVZ8O/qq8st/Fh0L8FWWk3mH7gP6wo083aUMC//ACFOIMFwL8=","dtype":"float64","order":"little","shape":[13]},"price_change_percentage_7d":{"__ndarray__":"U1k8q3mQ3z9yoPpI+ebtPzAB2dP2AQBAgmkOMZk89T8FNulI+JMEQCZRaSLGIPg/84wjaB1l1T8rMp3jf678P2VmURqKqeY/XK6ZksNx/T9DfMqy1o7dPyTPDB1Xu+0/srW7TleV4j8=","dtype":"float64","order":"little","shape":[13]}},"selected":{"id":"1771"},"selection_policy":{"id":"1791"}},"id":"1770","type":"ColumnDataSource"},{"attributes":{"label":{"value":"2"},"renderers":[{"id":"1752"}]},"id":"1768","type":"LegendItem"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#e5ae38"},"line_alpha":{"value":0.2},"line_color":{"value":"#e5ae38"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"1751","type":"Scatter"},{"attributes":{"axis_label":"price_change_percentage_24h","coordinates":null,"formatter":{"id":"1698"},"group":null,"major_label_policy":{"id":"1699"},"ticker":{"id":"1677"}},"id":"1676","type":"LinearAxis"},{"attributes":{"source":{"id":"1746"}},"id":"1753","type":"CDSView"},{"attributes":{},"id":"1686","type":"WheelZoomTool"},{"attributes":{},"id":"1685","type":"PanTool"},{"attributes":{"fill_color":{"value":"#6d904f"},"hatch_color":{"value":"#6d904f"},"line_color":{"value":"#6d904f"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"1773","type":"Scatter"},{"attributes":{"axis_label":"price_change_percentage_7d","coordinates":null,"formatter":{"id":"1701"},"group":null,"major_label_policy":{"id":"1702"},"ticker":{"id":"1681"}},"id":"1680","type":"LinearAxis"},{"attributes":{"axis":{"id":"1680"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"1683","type":"Grid"},{"attributes":{},"id":"1765","type":"UnionRenderers"},{"attributes":{},"id":"1681","type":"BasicTicker"},{"attributes":{"fill_color":{"value":"#fc4f30"},"hatch_color":{"value":"#fc4f30"},"line_color":{"value":"#fc4f30"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"1727","type":"Scatter"},{"attributes":{},"id":"1684","type":"SaveTool"},{"attributes":{"overlay":{"id":"1689"}},"id":"1687","type":"BoxZoomTool"},{"attributes":{"label":{"value":"1"},"renderers":[{"id":"1730"}]},"id":"1744","type":"LegendItem"},{"attributes":{},"id":"1688","type":"ResetTool"},{"attributes":{"source":{"id":"1724"}},"id":"1731","type":"CDSView"},{"attributes":{"coordinates":null,"data_source":{"id":"1724"},"glyph":{"id":"1727"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1729"},"nonselection_glyph":{"id":"1728"},"selection_glyph":{"id":"1745"},"view":{"id":"1731"}},"id":"1730","type":"GlyphRenderer"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"1689","type":"BoxAnnotation"},{"attributes":{},"id":"1718","type":"UnionRenderers"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#fc4f30"},"line_alpha":{"value":0.2},"line_color":{"value":"#fc4f30"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"1729","type":"Scatter"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#fc4f30"},"line_alpha":{"value":0.1},"line_color":{"value":"#fc4f30"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"1728","type":"Scatter"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.1},"line_color":{"value":"#30a2da"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"1707","type":"Scatter"},{"attributes":{},"id":"1725","type":"Selection"},{"attributes":{"fill_color":{"value":"#30a2da"},"hatch_color":{"value":"#30a2da"},"line_color":{"value":"#30a2da"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"1706","type":"Scatter"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#6d904f"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#6d904f"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#6d904f"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"circle"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"1795","type":"Scatter"},{"attributes":{},"id":"1741","type":"UnionRenderers"},{"attributes":{"data":{"CryptoCluster":[1],"coin_id":["ethlend"],"price_change_percentage_24h":{"__ndarray__":"3WeHPpbsE8A=","dtype":"float64","order":"little","shape":[1]},"price_change_percentage_7d":{"__ndarray__":"dIWi2pshp78=","dtype":"float64","order":"little","shape":[1]}},"selected":{"id":"1725"},"selection_policy":{"id":"1741"}},"id":"1724","type":"ColumnDataSource"},{"attributes":{"coordinates":null,"data_source":{"id":"1703"},"glyph":{"id":"1706"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"1708"},"nonselection_glyph":{"id":"1707"},"selection_glyph":{"id":"1723"},"view":{"id":"1710"}},"id":"1709","type":"GlyphRenderer"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#e5ae38"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#e5ae38"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"circle"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"1769","type":"Scatter"},{"attributes":{"tools":[{"id":"1663"},{"id":"1684"},{"id":"1685"},{"id":"1686"},{"id":"1687"},{"id":"1688"}]},"id":"1690","type":"Toolbar"},{"attributes":{},"id":"1699","type":"AllLabels"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#6d904f"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#6d904f"},"line_alpha":{"value":0.1},"line_color":{"value":"#6d904f"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"1774","type":"Scatter"},{"attributes":{},"id":"1698","type":"BasicTickFormatter"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer03144","sizing_mode":"stretch_width"},"id":"1660","type":"Spacer"},{"attributes":{},"id":"1701","type":"BasicTickFormatter"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.2},"line_color":{"value":"#30a2da"},"size":{"value":5.477225575051661},"x":{"field":"price_change_percentage_24h"},"y":{"field":"price_change_percentage_7d"}},"id":"1708","type":"Scatter"},{"attributes":{"children":[{"id":"1660"},{"id":"1667"},{"id":"1940"}],"margin":[0,0,0,0],"name":"Row03140","tags":["embedded"]},"id":"1659","type":"Row"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer03145","sizing_mode":"stretch_width"},"id":"1940","type":"Spacer"},{"attributes":{"data":{"CryptoCluster":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"coin_id":["tether","ripple","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","tron","okb","stellar","cdai","neo","leo-token","huobi-token","nem","binance-usd","iota","vechain","theta-token","dash","ethereum-classic","havven","omisego","ontology","ftx-token","true-usd","digibyte"],"price_change_percentage_24h":{"__ndarray__":"mCJF1OlLlj/mJRG3Ed+kv1qZGRifu9w//rJXHrAz1T/yMTqenpahPx9kk3dK7sM/GTRehHalwD90fnl54Irtv9XPDahDw9G/+VOWQSEmxz+kpbukw1bSP6eM4M4OgKo/BcMOj9+jqr/Chq6S5ebLv2h3TIfCZ68/weEQ8QyV0D/KmZTmC7niP1YH9WGFy/m/N7JuVBEB07/EU4GAhEGyv8Ocf0hw2/a/lrPBJY23/j/IKcQzxy3av5WrmDyeito/Bf22eE/6sz9Y362Ir3rzPw==","dtype":"float64","order":"little","shape":[26]},"price_change_percentage_7d":{"__ndarray__":"bZSIvk+a5r9RujxqH/Ppvw2U7adXaMi/d+r7VGbW+b9EGSx/8nTnv4rfBo4Lhe2/BEBwXEEApb8Udj8VbP/2vySelkVDp9i/7SZvVsyO5r8uu+JVH+LUvziN+gRMte2/o2qlez1D3b/CtPLKIS7rvyYjaswHnea/US6joeTvzz/dRmuCvtDvv/9TV66U6fq/ZWX+mWVCuD8GJ0pHuV/Nv8nIE2ZWH5q/TTxj/me11z+Htdvf1gXtv2Avihmzf9o/e31xtwIC5r/STlu6Y3Ljvw==","dtype":"float64","order":"little","shape":[26]}},"selected":{"id":"1704"},"selection_policy":{"id":"1718"}},"id":"1703","type":"ColumnDataSource"},{"attributes":{},"id":"1747","type":"Selection"},{"attributes":{"end":2.997678656273595,"reset_end":2.997678656273595,"reset_start":-2.107454305728652,"start":-2.107454305728652,"tags":[[["price_change_percentage_7d","price_change_percentage_7d",null]]]},"id":"1662","type":"Range1d"},{"attributes":{},"id":"1702","type":"AllLabels"},{"attributes":{"end":2.2155632386560065,"reset_end":2.2155632386560065,"reset_start":-5.276792781891412,"start":-5.276792781891412,"tags":[[["price_change_percentage_24h","price_change_percentage_24h",null]]]},"id":"1661","type":"Range1d"},{"attributes":{},"id":"1704","type":"Selection"}],"root_ids":["1659"]},"title":"Bokeh Application","version":"2.4.1"}};
    var render_items = [{"docid":"e05261df-dd91-44b5-b7cd-f19891b15873","root_ids":["1659"],"roots":{"1659":"8d2d71c9-5896-478e-9bb7-93d9614facda"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>



---

### Optimize Clusters with Principal Component Analysis

In this section, you will perform a principal component analysis (PCA) and reduce the features to three principal components.

1. Create a PCA model instance and set `n_components=3`.

2. Use the PCA model to reduce to three principal components. View the first five rows of the DataFrame. 

3. Retrieve the explained variance to determine how much information can be attributed to each principal component.

4. Answer the following question: What is the total explained variance of the three principal components?

5. Create a new DataFrame with the PCA data. Be sure to set the `coin_id` index from the original DataFrame as the index for the new DataFrame. Review the resulting DataFrame.


```python
# Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components = 3)
```


```python
# Use the PCA model with `fit_transform` to reduce to 
# three principal components.
crypto_pca_data = pca.fit_transform(df_market_data_scaled)

# View the first five rows of the DataFrame. 
crypto_pca_data[:5]
```




    array([[-0.60066733,  0.84276006,  0.46159457],
           [-0.45826071,  0.45846566,  0.95287678],
           [-0.43306981, -0.16812638, -0.64175193],
           [-0.47183495, -0.22266008, -0.47905316],
           [-1.15779997,  2.04120919,  1.85971527]])




```python
# Retrieve the explained variance to determine how much information 
# can be attributed to each principal component.
pca.explained_variance_ratio_
```




    array([0.3719856 , 0.34700813, 0.17603793])




```python
crypto_pca_data
```




    array([[-0.60066733,  0.84276006,  0.46159457],
           [-0.45826071,  0.45846566,  0.95287678],
           [-0.43306981, -0.16812638, -0.64175193],
           [-0.47183495, -0.22266008, -0.47905316],
           [-1.15779997,  2.04120919,  1.85971527],
           [-0.51653377,  1.38837748,  0.80407131],
           [-0.45071134,  0.51769912,  2.84614316],
           [-0.34559977,  0.72943939,  1.47801284],
           [-0.64946792,  0.43216514,  0.60030286],
           [-0.75901394, -0.20119979, -0.21765292],
           [-0.24819846, -1.37625159, -1.46202571],
           [-0.43840762, -0.17533654, -0.6633884 ],
           [-0.69342533, -0.47381462, -0.52759693],
           [ 0.06049915,  2.90940385,  1.49857131],
           [-0.39335243, -0.10819197, -0.01275608],
           [-0.79617564, -0.49440875,  1.08281169],
           [ 0.06407452, -1.26982514, -1.09882928],
           [-0.48901506, -0.73271912, -0.06254323],
           [-0.3062723 ,  0.70341515,  1.71422359],
           [-0.51352775, -0.14280239, -0.65656583],
           [-0.36212044, -0.98691441, -0.72875232],
           [-0.60426463,  0.82739764,  0.43931594],
           [-0.4132956 , -0.67411527, -1.07662834],
           [-0.40748304, -0.21250655, -0.35142563],
           [ 0.60897382,  0.56353212, -1.14874159],
           [-0.45021114, -0.15101945, -0.64740061],
           [-0.76466522, -0.51788554,  0.20499029],
           [-0.55631468, -1.93820906, -1.26177589],
           [-0.42514677,  0.49297617,  1.05804837],
           [ 2.67686761, -0.0139541 , -1.96520722],
           [-0.61392275, -0.4793368 ,  0.33956513],
           [-0.57992398, -0.35633377, -0.11494202],
           [ 8.08901821, -3.89689054,  2.30138208],
           [-0.38904526,  0.16504063,  0.3794137 ],
           [ 0.86576183, -2.26188239,  0.27558289],
           [ 0.11167508,  0.42831576, -1.20539797],
           [ 4.7923954 ,  6.76767868, -1.98698545],
           [-0.63235492, -2.10811713, -0.65222738],
           [-0.59314216,  0.02148496,  0.20991142],
           [-0.4581305 , -0.13573403, -0.63528357],
           [-0.29791045, -0.1911256 , -0.90960173]])



#### Answer the following question: What is the total explained variance of the three principal components?

**Question:** What is the total explained variance of the three principal components?

**Answer:** Variance is the spread between numbers in a dataset, variance itself meaning how far away numbers are from the mean.  So the total explained variance would be the sum of the three explained variance amounts divided by the total variance.  That would be .29603494


```python
# Create a new DataFrame with the PCA data.
# Note: The code for this step is provided for you

# Creating a DataFrame with the PCA data
df_crypto_pca = pd.DataFrame(crypto_pca_data, columns=["PC1", "PC2", "PC3"])

# Copy the crypto names from the original data
df_crypto_pca["coin_id"] = df_market_data.index

# Set the coinid column as index
df_crypto_pca = df_crypto_pca.set_index("coin_id")

# Display sample data
df_crypto_pca.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
    </tr>
    <tr>
      <th>coin_id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bitcoin</th>
      <td>-0.600667</td>
      <td>0.842760</td>
      <td>0.461595</td>
    </tr>
    <tr>
      <th>ethereum</th>
      <td>-0.458261</td>
      <td>0.458466</td>
      <td>0.952877</td>
    </tr>
    <tr>
      <th>tether</th>
      <td>-0.433070</td>
      <td>-0.168126</td>
      <td>-0.641752</td>
    </tr>
    <tr>
      <th>ripple</th>
      <td>-0.471835</td>
      <td>-0.222660</td>
      <td>-0.479053</td>
    </tr>
    <tr>
      <th>bitcoin-cash</th>
      <td>-1.157800</td>
      <td>2.041209</td>
      <td>1.859715</td>
    </tr>
  </tbody>
</table>
</div>



---

### Find the Best Value for k Using the PCA Data

In this section, you will use the elbow method to find the best value for `k` using the PCA data.

1. Code the elbow method algorithm and use the PCA data to find the best value for `k`. Use a range from 1 to 11. 

2. Plot a line chart with all the inertia values computed with the different values of `k` to visually identify the optimal value for `k`.

3. Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?


```python
# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))
```


```python
# Create an empy list to store the inertia values
inertia = []
```


```python
# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_market_data_pca`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    model = KMeans(n_clusters=i, random_state=0)
    model.fit(df_crypto_pca)
    inertia.append(model.inertia_)
    
inertia
```

    C:\Users\ijisk\anaconda3\lib\site-packages\sklearn\cluster\_kmeans.py:881: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
      warnings.warn(
    




    [256.87408556789256,
     165.9019940203602,
     93.77462568057304,
     49.66549665179738,
     37.87874703346251,
     27.61897178795745,
     21.182775862957335,
     17.389823204768913,
     13.593650379876742,
     10.559357562793437]




```python
# Create a dictionary with the data to plot the Elbow curve
elbow_data_2 = { 
    "k": k,
    "inertia": inertia
}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow_data_2 = pd.DataFrame(elbow_data_2)
```


```python
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
df_elbow_data_2.hvplot.line(
    x="k",
    y="inertia",
    title="Elbow Curve",
    xticks=k
)
```






<div id='2022'>





  <div class="bk-root" id="845dfd75-347b-4145-a509-7c1f61309139" data-root-id="2022"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"4bd1180d-7f51-420b-8190-5bcaef236456":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{"axis_label":"k","coordinates":null,"formatter":{"id":"2068"},"group":null,"major_label_policy":{"id":"2069"},"ticker":{"id":"2066"}},"id":"2036","type":"LinearAxis"},{"attributes":{},"id":"2058","type":"Selection"},{"attributes":{},"id":"2045","type":"PanTool"},{"attributes":{"axis_label":"inertia","coordinates":null,"formatter":{"id":"2073"},"group":null,"major_label_policy":{"id":"2074"},"ticker":{"id":"2041"}},"id":"2040","type":"LinearAxis"},{"attributes":{"coordinates":null,"data_source":{"id":"2057"},"glyph":{"id":"2060"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2062"},"nonselection_glyph":{"id":"2061"},"selection_glyph":{"id":"2065"},"view":{"id":"2064"}},"id":"2063","type":"GlyphRenderer"},{"attributes":{},"id":"2041","type":"BasicTicker"},{"attributes":{},"id":"2068","type":"BasicTickFormatter"},{"attributes":{},"id":"2046","type":"WheelZoomTool"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2060","type":"Line"},{"attributes":{},"id":"2044","type":"SaveTool"},{"attributes":{"line_alpha":0.1,"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2061","type":"Line"},{"attributes":{"overlay":{"id":"2049"}},"id":"2047","type":"BoxZoomTool"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2065","type":"Line"},{"attributes":{},"id":"2048","type":"ResetTool"},{"attributes":{"line_alpha":0.2,"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"2062","type":"Line"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer03862","sizing_mode":"stretch_width"},"id":"2093","type":"Spacer"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"2049","type":"BoxAnnotation"},{"attributes":{"source":{"id":"2057"}},"id":"2064","type":"CDSView"},{"attributes":{"ticks":[1,2,3,4,5,6,7,8,9,10]},"id":"2066","type":"FixedTicker"},{"attributes":{},"id":"2034","type":"LinearScale"},{"attributes":{},"id":"2074","type":"AllLabels"},{"attributes":{},"id":"2069","type":"AllLabels"},{"attributes":{},"id":"2032","type":"LinearScale"},{"attributes":{},"id":"2081","type":"UnionRenderers"},{"attributes":{"children":[{"id":"2023"},{"id":"2027"},{"id":"2093"}],"margin":[0,0,0,0],"name":"Row03857","tags":["embedded"]},"id":"2022","type":"Row"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer03861","sizing_mode":"stretch_width"},"id":"2023","type":"Spacer"},{"attributes":{"callback":null,"renderers":[{"id":"2063"}],"tags":["hv_created"],"tooltips":[["k","@{k}"],["inertia","@{inertia}"]]},"id":"2026","type":"HoverTool"},{"attributes":{"end":10.0,"reset_end":10.0,"reset_start":1.0,"start":1.0,"tags":[[["k","k",null]]]},"id":"2024","type":"Range1d"},{"attributes":{"axis":{"id":"2036"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"2039","type":"Grid"},{"attributes":{"below":[{"id":"2036"}],"center":[{"id":"2039"},{"id":"2043"}],"height":300,"left":[{"id":"2040"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"2063"}],"sizing_mode":"fixed","title":{"id":"2028"},"toolbar":{"id":"2050"},"width":700,"x_range":{"id":"2024"},"x_scale":{"id":"2032"},"y_range":{"id":"2025"},"y_scale":{"id":"2034"}},"id":"2027","subtype":"Figure","type":"Plot"},{"attributes":{"end":281.5055583684025,"reset_end":281.5055583684025,"reset_start":-14.072115237716478,"start":-14.072115237716478,"tags":[[["inertia","inertia",null]]]},"id":"2025","type":"Range1d"},{"attributes":{"tools":[{"id":"2026"},{"id":"2044"},{"id":"2045"},{"id":"2046"},{"id":"2047"},{"id":"2048"}]},"id":"2050","type":"Toolbar"},{"attributes":{},"id":"2073","type":"BasicTickFormatter"},{"attributes":{"axis":{"id":"2040"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"2043","type":"Grid"},{"attributes":{"data":{"inertia":{"__ndarray__":"EQAmQfwNcEBPVJAi3bxkQP0sl3eTcVdAnIiJ/i7VSEDZFmXIevBCQIRsYu90njtAYuYhZsouNUBTtBt0y2MxQDyoTfHyLytAqO9NHWQeJUA=","dtype":"float64","order":"little","shape":[10]},"k":[1,2,3,4,5,6,7,8,9,10]},"selected":{"id":"2058"},"selection_policy":{"id":"2081"}},"id":"2057","type":"ColumnDataSource"},{"attributes":{"coordinates":null,"group":null,"text":"Elbow Curve","text_color":"black","text_font_size":"12pt"},"id":"2028","type":"Title"}],"root_ids":["2022"]},"title":"Bokeh Application","version":"2.4.1"}};
    var render_items = [{"docid":"4bd1180d-7f51-420b-8190-5bcaef236456","root_ids":["2022"],"roots":{"2022":"845dfd75-347b-4145-a509-7c1f61309139"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>



#### Answer the following questions: What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?
* **Question:** What is the best value for `k` when using the PCA data?

  * **Answer:** The best value for k is 4


* **Question:** Does it differ from the best k value found using the original data?

  * **Answer:** No it doesn't differ from the best value using the original data.

---

### Cluster Cryptocurrencies with K-means Using the PCA Data

In this section, you will use the PCA data and the K-Means algorithm with the best value for `k` found in the previous section to cluster the cryptocurrencies according to the principal components.

1. Initialize the K-Means model with four clusters using the best value for `k`. 

2. Fit the K-Means model using the PCA data.

3. Predict the clusters to group the cryptocurrencies using the PCA data. View the resulting array of cluster values.

4. Add a new column to the DataFrame with the PCA data to store the predicted clusters.

5. Create a scatter plot using hvPlot by setting `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. Color the graph points with the labels found using K-Means and add the crypto name in the `hover_cols` parameter to identify the cryptocurrency represented by each data point.


```python
# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4)
```


```python
# Fit the K-Means model using the PCA data
model.fit(df_crypto_pca)
```




    KMeans(n_clusters=4)




```python
# Predict the clusters to group the cryptocurrencies using the PCA data
df_crypto_clusters = model.predict(df_crypto_pca)

# View the resulting array of cluster values.
df_crypto_clusters
```




    array([0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0,
           1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 2, 0, 1, 1, 3, 1, 1, 1, 1])




```python
# Create a copy of the DataFrame with the PCA data
df_crypto_scaled_new = df_crypto_pca.copy()

# Add a new column to the DataFrame with the predicted clusters
df_crypto_scaled_new["CryptoCluster"] = df_crypto_clusters

# Display sample data
df_crypto_scaled_new.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>CryptoCluster</th>
    </tr>
    <tr>
      <th>coin_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bitcoin</th>
      <td>-0.600667</td>
      <td>0.842760</td>
      <td>0.461595</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ethereum</th>
      <td>-0.458261</td>
      <td>0.458466</td>
      <td>0.952877</td>
      <td>0</td>
    </tr>
    <tr>
      <th>tether</th>
      <td>-0.433070</td>
      <td>-0.168126</td>
      <td>-0.641752</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ripple</th>
      <td>-0.471835</td>
      <td>-0.222660</td>
      <td>-0.479053</td>
      <td>1</td>
    </tr>
    <tr>
      <th>bitcoin-cash</th>
      <td>-1.157800</td>
      <td>2.041209</td>
      <td>1.859715</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a scatter plot using hvPlot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`. 
# Color the graph points with the labels found using K-Means and 
# add the crypto name in the `hover_cols` parameter to identify 
# the cryptocurrency represented by each data point.
df_crypto_scaled_new.hvplot.scatter(
   # x="price_change_percentage_24h",
   # y="price_change_percentage_7d",
    x="PC1",
    y="PC2",
    by="CryptoCluster",
    hover_cols = ["coin_id"],
    title="Scatter Plot by Crypto Segment - k4"
)
```






<div id='2140'>





  <div class="bk-root" id="8f4a25ae-28ac-4c18-a8f2-44f15eae85d0" data-root-id="2140"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"24e72f91-57a6-4e11-98bf-eaa711446007":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{},"id":"2222","type":"UnionRenderers"},{"attributes":{},"id":"2199","type":"UnionRenderers"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#e5ae38"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#e5ae38"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"circle"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2250","type":"Scatter"},{"attributes":{"label":{"value":"3"},"renderers":[{"id":"2257"}]},"id":"2275","type":"LegendItem"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer04113","sizing_mode":"stretch_width"},"id":"2421","type":"Spacer"},{"attributes":{"coordinates":null,"data_source":{"id":"2251"},"glyph":{"id":"2254"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2256"},"nonselection_glyph":{"id":"2255"},"selection_glyph":{"id":"2276"},"view":{"id":"2258"}},"id":"2257","type":"GlyphRenderer"},{"attributes":{"fill_color":{"value":"#30a2da"},"hatch_color":{"value":"#30a2da"},"line_color":{"value":"#30a2da"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2187","type":"Scatter"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#6d904f"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#6d904f"},"line_alpha":{"value":0.1},"line_color":{"value":"#6d904f"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2255","type":"Scatter"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer04112","sizing_mode":"stretch_width"},"id":"2141","type":"Spacer"},{"attributes":{},"id":"2180","type":"AllLabels"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#fc4f30"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#fc4f30"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"circle"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2226","type":"Scatter"},{"attributes":{},"id":"2179","type":"BasicTickFormatter"},{"attributes":{},"id":"2183","type":"AllLabels"},{"attributes":{"click_policy":"mute","coordinates":null,"group":null,"items":[{"id":"2203"},{"id":"2225"},{"id":"2249"},{"id":"2275"}],"location":[0,0],"title":"CryptoCluster"},"id":"2202","type":"Legend"},{"attributes":{"children":[{"id":"2141"},{"id":"2148"},{"id":"2421"}],"margin":[0,0,0,0],"name":"Row04108","tags":["embedded"]},"id":"2140","type":"Row"},{"attributes":{"coordinates":null,"data_source":{"id":"2184"},"glyph":{"id":"2187"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2189"},"nonselection_glyph":{"id":"2188"},"selection_glyph":{"id":"2204"},"view":{"id":"2191"}},"id":"2190","type":"GlyphRenderer"},{"attributes":{},"id":"2153","type":"LinearScale"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.1},"line_color":{"value":"#30a2da"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2188","type":"Scatter"},{"attributes":{"label":{"value":"2"},"renderers":[{"id":"2233"}]},"id":"2249","type":"LegendItem"},{"attributes":{"coordinates":null,"group":null,"text":"Scatter Plot by Crypto Segment - k4","text_color":"black","text_font_size":"12pt"},"id":"2149","type":"Title"},{"attributes":{},"id":"2182","type":"BasicTickFormatter"},{"attributes":{"fill_color":{"value":"#e5ae38"},"hatch_color":{"value":"#e5ae38"},"line_color":{"value":"#e5ae38"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2230","type":"Scatter"},{"attributes":{"source":{"id":"2251"}},"id":"2258","type":"CDSView"},{"attributes":{},"id":"2252","type":"Selection"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#30a2da"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#30a2da"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"circle"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2204","type":"Scatter"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#6d904f"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#6d904f"},"line_alpha":{"value":0.2},"line_color":{"value":"#6d904f"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2256","type":"Scatter"},{"attributes":{},"id":"2272","type":"UnionRenderers"},{"attributes":{"end":8.485310422788737,"reset_end":8.485310422788737,"reset_start":-1.5540921804637602,"start":-1.5540921804637602,"tags":[[["PC1","PC1",null]]]},"id":"2142","type":"Range1d"},{"attributes":{"coordinates":null,"data_source":{"id":"2227"},"glyph":{"id":"2230"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2232"},"nonselection_glyph":{"id":"2231"},"selection_glyph":{"id":"2250"},"view":{"id":"2234"}},"id":"2233","type":"GlyphRenderer"},{"attributes":{"fill_color":{"value":"#6d904f"},"hatch_color":{"value":"#6d904f"},"line_color":{"value":"#6d904f"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2254","type":"Scatter"},{"attributes":{"callback":null,"renderers":[{"id":"2190"},{"id":"2211"},{"id":"2233"},{"id":"2257"}],"tags":["hv_created"],"tooltips":[["CryptoCluster","@{CryptoCluster}"],["PC1","@{PC1}"],["PC2","@{PC2}"],["coin_id","@{coin_id}"]]},"id":"2144","type":"HoverTool"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#e5ae38"},"line_alpha":{"value":0.1},"line_color":{"value":"#e5ae38"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2231","type":"Scatter"},{"attributes":{"label":{"value":"1"},"renderers":[{"id":"2211"}]},"id":"2225","type":"LegendItem"},{"attributes":{},"id":"2185","type":"Selection"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.2},"line_color":{"value":"#30a2da"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2189","type":"Scatter"},{"attributes":{"end":7.83413559633714,"reset_end":7.83413559633714,"reset_start":-4.963347456561368,"start":-4.963347456561368,"tags":[[["PC2","PC2",null]]]},"id":"2143","type":"Range1d"},{"attributes":{"below":[{"id":"2157"}],"center":[{"id":"2160"},{"id":"2164"}],"height":300,"left":[{"id":"2161"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"2190"},{"id":"2211"},{"id":"2233"},{"id":"2257"}],"right":[{"id":"2202"}],"sizing_mode":"fixed","title":{"id":"2149"},"toolbar":{"id":"2171"},"width":700,"x_range":{"id":"2142"},"x_scale":{"id":"2153"},"y_range":{"id":"2143"},"y_scale":{"id":"2155"}},"id":"2148","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"2155","type":"LinearScale"},{"attributes":{"data":{"CryptoCluster":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],"PC1":{"__ndarray__":"TJZ8amq3278UYXw2izLev9jya5vXSei/S/tzoPfEz7+Bh/TX3g7cv4oZGVOKMOa/gkwjp68s2b/0BLsJMGewP7SyldMFTN+/DekqwdFu4L/bFokx+yzXv5MSO2dvc9q/xxEoujMU2r94ZdmqtnzjP+hN1mNC0Ny/iti9MiN46L/kse9uVM3hv+hGFZE5agVAmSC8UEGl47+1vuS6vI7iv3QKKydStOs/cJK6/7yWvD9vXMBgQDzkv5YbmkYF++K/Dr1JmQJS3b/84a/49hDTvw==","dtype":"float64","order":"little","shape":[26]},"PC2":{"__ndarray__":"lNF7TyqFxb+vj1UaIIDMv6PhzCbqwMm/yzoUYyAF9r+8ZsaEbXHGv0fJIJL6Ut6/fyu0IHiyu79tgFApNFH0vwWhJl5vcue/1JKaSllHwr+6iEuGzZTvv2r5wy5akuW/tN3jHmozy7+sF2SFdAjiP2kuIf2aVMO/NOuFsYSS4L9ZbbiA5wL/vxN2Joz3k4y/jY8PRXSt3r8M37YoLM7Wv2X+1spVGALA5CEohoZp2z9DqFKDbN0AwPUO7msnAJY/VfWtmbtfwb9ISPDBzXbIvw==","dtype":"float64","order":"little","shape":[26]},"coin_id":["tether","ripple","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","tron","okb","stellar","cdai","neo","leo-token","huobi-token","nem","binance-usd","iota","vechain","theta-token","dash","ethereum-classic","havven","omisego","ontology","ftx-token","true-usd","digibyte"]},"selected":{"id":"2206"},"selection_policy":{"id":"2222"}},"id":"2205","type":"ColumnDataSource"},{"attributes":{"data":{"CryptoCluster":[3],"PC1":{"__ndarray__":"RxQ7s2krE0A=","dtype":"float64","order":"little","shape":[1]},"PC2":{"__ndarray__":"JVrQWxoSG0A=","dtype":"float64","order":"little","shape":[1]},"coin_id":["celsius-degree-token"]},"selected":{"id":"2252"},"selection_policy":{"id":"2272"}},"id":"2251","type":"ColumnDataSource"},{"attributes":{"label":{"value":"0"},"renderers":[{"id":"2190"}]},"id":"2203","type":"LegendItem"},{"attributes":{"coordinates":null,"data_source":{"id":"2205"},"glyph":{"id":"2208"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2210"},"nonselection_glyph":{"id":"2209"},"selection_glyph":{"id":"2226"},"view":{"id":"2212"}},"id":"2211","type":"GlyphRenderer"},{"attributes":{"source":{"id":"2227"}},"id":"2234","type":"CDSView"},{"attributes":{"source":{"id":"2184"}},"id":"2191","type":"CDSView"},{"attributes":{},"id":"2158","type":"BasicTicker"},{"attributes":{"axis":{"id":"2157"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"2160","type":"Grid"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#e5ae38"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#e5ae38"},"line_alpha":{"value":0.2},"line_color":{"value":"#e5ae38"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2232","type":"Scatter"},{"attributes":{},"id":"2246","type":"UnionRenderers"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"2170","type":"BoxAnnotation"},{"attributes":{"axis_label":"PC1","coordinates":null,"formatter":{"id":"2179"},"group":null,"major_label_policy":{"id":"2180"},"ticker":{"id":"2158"}},"id":"2157","type":"LinearAxis"},{"attributes":{"fill_color":{"value":"#fc4f30"},"hatch_color":{"value":"#fc4f30"},"line_color":{"value":"#fc4f30"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2208","type":"Scatter"},{"attributes":{},"id":"2228","type":"Selection"},{"attributes":{},"id":"2167","type":"WheelZoomTool"},{"attributes":{},"id":"2166","type":"PanTool"},{"attributes":{"data":{"CryptoCluster":[0,0,0,0,0,0,0,0,0,0,0,0,0],"PC1":{"__ndarray__":"S/XQrqo447/f0e2+JFTdv2ATj0NZhvK/ItMJ1HGH4L/0WkdgdNjcvwn+x3tOHta/lm1M83DI5L/osk+KvvmuP5kxxVNFeum/186LH/eZ07/b3W/HIlbjv2Hz5tCaNdu/vEmjFh7m2L8=","dtype":"float64","order":"little","shape":[13]},"PC2":{"__ndarray__":"tQJY8+P36j9hP81egFfdP6JEN3tlVABAbaRxTss29j+zZPC8/ZDgP3/E/UiRV+c/WS0795eo2z+X9tOGdUYHQHCknptkpN+/aENue2CC5j9WCxibCnrqP9N0su7rjN8/T0ntLg0gxT8=","dtype":"float64","order":"little","shape":[13]},"coin_id":["bitcoin","ethereum","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","monero","tezos","cosmos","wrapped-bitcoin","zcash","maker"]},"selected":{"id":"2185"},"selection_policy":{"id":"2199"}},"id":"2184","type":"ColumnDataSource"},{"attributes":{"data":{"CryptoCluster":[2],"PC1":{"__ndarray__":"TlWiy5MtIEA=","dtype":"float64","order":"little","shape":[1]},"PC2":{"__ndarray__":"IQ3w8dQsD8A=","dtype":"float64","order":"little","shape":[1]},"coin_id":["ethlend"]},"selected":{"id":"2228"},"selection_policy":{"id":"2246"}},"id":"2227","type":"ColumnDataSource"},{"attributes":{"axis_label":"PC2","coordinates":null,"formatter":{"id":"2182"},"group":null,"major_label_policy":{"id":"2183"},"ticker":{"id":"2162"}},"id":"2161","type":"LinearAxis"},{"attributes":{"tools":[{"id":"2144"},{"id":"2165"},{"id":"2166"},{"id":"2167"},{"id":"2168"},{"id":"2169"}]},"id":"2171","type":"Toolbar"},{"attributes":{"axis":{"id":"2161"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"2164","type":"Grid"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#fc4f30"},"line_alpha":{"value":0.1},"line_color":{"value":"#fc4f30"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2209","type":"Scatter"},{"attributes":{"source":{"id":"2205"}},"id":"2212","type":"CDSView"},{"attributes":{},"id":"2162","type":"BasicTicker"},{"attributes":{},"id":"2165","type":"SaveTool"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#fc4f30"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#fc4f30"},"line_alpha":{"value":0.2},"line_color":{"value":"#fc4f30"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2210","type":"Scatter"},{"attributes":{"overlay":{"id":"2170"}},"id":"2168","type":"BoxZoomTool"},{"attributes":{},"id":"2169","type":"ResetTool"},{"attributes":{},"id":"2206","type":"Selection"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#6d904f"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#6d904f"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#6d904f"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"circle"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2276","type":"Scatter"}],"root_ids":["2140"]},"title":"Bokeh Application","version":"2.4.1"}};
    var render_items = [{"docid":"24e72f91-57a6-4e11-98bf-eaa711446007","root_ids":["2140"],"roots":{"2140":"8f4a25ae-28ac-4c18-a8f2-44f15eae85d0"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>



---

### Visualize and Compare the Results

In this section, you will visually analyze the cluster analysis results by contrasting the outcome with and without using the optimization techniques.

1. Create a composite plot using hvPlot and the plus (`+`) operator to contrast the Elbow Curve that you created to find the best value for `k` with the original and the PCA data.

2. Create a composite plot using hvPlot and the plus (`+`) operator to contrast the cryptocurrencies clusters using the original and the PCA data.

3. Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?

> **Rewind:** Back in Lesson 3 of Module 6, you learned how to create composite plots. You can look at that lesson to review how to make these plots; also, you can check [the hvPlot documentation](https://holoviz.org/tutorial/Composing_Plots.html).


```python
#Composite of df_elbow_data_1 + df_elbow_data_2:
composed = df_elbow_data_1 + df_elbow_data_2
```


```python
# Composite visualization plot to contrast the Elbow curves
composed.hvplot.line(
    x="k",
    y="inertia",
    title="Composite Visualization of the Elbow curves of Inertia and k"
)

```






<div id='3392'>





  <div class="bk-root" id="1148e788-9f59-439e-8d61-8967c7682b80" data-root-id="3392"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"53755995-1bbe-42b2-b1d6-7eeac94df93e":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{},"id":"3418","type":"ResetTool"},{"attributes":{},"id":"3437","type":"BasicTickFormatter"},{"attributes":{},"id":"3449","type":"UnionRenderers"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"3435","type":"Line"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"3419","type":"BoxAnnotation"},{"attributes":{},"id":"3402","type":"LinearScale"},{"attributes":{"tools":[{"id":"3396"},{"id":"3414"},{"id":"3415"},{"id":"3416"},{"id":"3417"},{"id":"3418"}]},"id":"3420","type":"Toolbar"},{"attributes":{},"id":"3428","type":"Selection"},{"attributes":{"line_alpha":0.2,"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"3432","type":"Line"},{"attributes":{"source":{"id":"3427"}},"id":"3434","type":"CDSView"},{"attributes":{"callback":null,"renderers":[{"id":"3433"}],"tags":["hv_created"],"tooltips":[["k","@{k}"],["inertia","@{inertia}"]]},"id":"3396","type":"HoverTool"},{"attributes":{},"id":"3441","type":"AllLabels"},{"attributes":{},"id":"3438","type":"AllLabels"},{"attributes":{"children":[{"id":"3393"},{"id":"3397"},{"id":"3462"}],"margin":[0,0,0,0],"name":"Row07081","tags":["embedded"]},"id":"3392","type":"Row"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer07086","sizing_mode":"stretch_width"},"id":"3462","type":"Spacer"},{"attributes":{"end":594.3832684393415,"reset_end":594.3832684393415,"reset_start":-11.72692601804664,"start":-11.72692601804664,"tags":[[["inertia","inertia",null]]]},"id":"3395","type":"Range1d"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer07085","sizing_mode":"stretch_width"},"id":"3393","type":"Spacer"},{"attributes":{"coordinates":null,"data_source":{"id":"3427"},"glyph":{"id":"3430"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"3432"},"nonselection_glyph":{"id":"3431"},"selection_glyph":{"id":"3435"},"view":{"id":"3434"}},"id":"3433","type":"GlyphRenderer"},{"attributes":{"end":20.0,"reset_end":20.0,"reset_start":2.0,"start":2.0,"tags":[[["k","k",null]]]},"id":"3394","type":"Range1d"},{"attributes":{"data":{"inertia":{"__ndarray__":"CACTIP7+gEDgbWEujpt2QG/fKSniHmtANNn5iQMWYEDUnD2Sl8tZQJSQXrd6IFRAXQ1W4zxGUECW1e/YE3RLQNDecxsZCkdAsDcZ/iBkQ0A=","dtype":"float64","order":"little","shape":[10]},"k":[2,4,6,8,10,12,14,16,18,20]},"selected":{"id":"3428"},"selection_policy":{"id":"3449"}},"id":"3427","type":"ColumnDataSource"},{"attributes":{},"id":"3407","type":"BasicTicker"},{"attributes":{},"id":"3404","type":"LinearScale"},{"attributes":{"coordinates":null,"group":null,"text":"Composite Visualization of the Elbow curves of Inertia and k","text_color":"black","text_font_size":"12pt"},"id":"3398","type":"Title"},{"attributes":{"axis":{"id":"3406"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"3409","type":"Grid"},{"attributes":{},"id":"3440","type":"BasicTickFormatter"},{"attributes":{"line_alpha":0.1,"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"3431","type":"Line"},{"attributes":{"axis_label":"k","coordinates":null,"formatter":{"id":"3437"},"group":null,"major_label_policy":{"id":"3438"},"ticker":{"id":"3407"}},"id":"3406","type":"LinearAxis"},{"attributes":{"line_color":"#30a2da","line_width":2,"x":{"field":"k"},"y":{"field":"inertia"}},"id":"3430","type":"Line"},{"attributes":{"below":[{"id":"3406"}],"center":[{"id":"3409"},{"id":"3413"}],"height":300,"left":[{"id":"3410"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"3433"}],"sizing_mode":"fixed","title":{"id":"3398"},"toolbar":{"id":"3420"},"width":700,"x_range":{"id":"3394"},"x_scale":{"id":"3402"},"y_range":{"id":"3395"},"y_scale":{"id":"3404"}},"id":"3397","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"3415","type":"PanTool"},{"attributes":{"axis":{"id":"3410"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"3413","type":"Grid"},{"attributes":{"axis_label":"inertia","coordinates":null,"formatter":{"id":"3440"},"group":null,"major_label_policy":{"id":"3441"},"ticker":{"id":"3411"}},"id":"3410","type":"LinearAxis"},{"attributes":{},"id":"3416","type":"WheelZoomTool"},{"attributes":{},"id":"3411","type":"BasicTicker"},{"attributes":{},"id":"3414","type":"SaveTool"},{"attributes":{"overlay":{"id":"3419"}},"id":"3417","type":"BoxZoomTool"}],"root_ids":["3392"]},"title":"Bokeh Application","version":"2.4.1"}};
    var render_items = [{"docid":"53755995-1bbe-42b2-b1d6-7eeac94df93e","root_ids":["3392"],"roots":{"3392":"1148e788-9f59-439e-8d61-8967c7682b80"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>




```python
# Compoosite visualization plot to contrast the clusters
df_cluster_composite = pd.merge(df_crypto_scaled_predictions, df_crypto_scaled_new, on=['coin_id'], how='inner')
df_cluster_composite.hvplot.scatter(
    y="PC2",
    x="PC1",
    hover_cols=["coin_id"],
    title="Contrast of Scatter plots with Original Data and PCA Data",
    xticks=k
)
```






<div id='2758'>





  <div class="bk-root" id="fa84e68a-160c-4433-99de-9110fb4e7b4d" data-root-id="2758"></div>
</div>
<script type="application/javascript">(function(root) {
  function embed_document(root) {
    var docs_json = {"4aaf8536-c5b4-479a-becc-2ab00418f977":{"defs":[{"extends":null,"module":null,"name":"ReactiveHTML1","overrides":[],"properties":[]},{"extends":null,"module":null,"name":"FlexBox1","overrides":[],"properties":[{"default":"flex-start","kind":null,"name":"align_content"},{"default":"flex-start","kind":null,"name":"align_items"},{"default":"row","kind":null,"name":"flex_direction"},{"default":"wrap","kind":null,"name":"flex_wrap"},{"default":"flex-start","kind":null,"name":"justify_content"}]},{"extends":null,"module":null,"name":"TemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]},{"extends":null,"module":null,"name":"MaterialTemplateActions1","overrides":[],"properties":[{"default":0,"kind":null,"name":"open_modal"},{"default":0,"kind":null,"name":"close_modal"}]}],"roots":{"references":[{"attributes":{"fill_color":{"value":"#30a2da"},"hatch_color":{"value":"#30a2da"},"line_color":{"value":"#30a2da"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2796","type":"Scatter"},{"attributes":{},"id":"2768","type":"LinearScale"},{"attributes":{},"id":"2784","type":"ResetTool"},{"attributes":{"coordinates":null,"data_source":{"id":"2793"},"glyph":{"id":"2796"},"group":null,"hover_glyph":null,"muted_glyph":{"id":"2798"},"nonselection_glyph":{"id":"2797"},"selection_glyph":{"id":"2801"},"view":{"id":"2800"}},"id":"2799","type":"GlyphRenderer"},{"attributes":{"source":{"id":"2793"}},"id":"2800","type":"CDSView"},{"attributes":{"bottom_units":"screen","coordinates":null,"fill_alpha":0.5,"fill_color":"lightgrey","group":null,"left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","syncable":false,"top_units":"screen"},"id":"2785","type":"BoxAnnotation"},{"attributes":{},"id":"2805","type":"AllLabels"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.2},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.2},"line_color":{"value":"#30a2da"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2798","type":"Scatter"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer05286","sizing_mode":"stretch_width"},"id":"2829","type":"Spacer"},{"attributes":{"ticks":[1,2,3,4,5,6,7,8,9,10]},"id":"2802","type":"FixedTicker"},{"attributes":{"margin":[5,5,5,5],"name":"HSpacer05285","sizing_mode":"stretch_width"},"id":"2759","type":"Spacer"},{"attributes":{},"id":"2770","type":"LinearScale"},{"attributes":{},"id":"2810","type":"AllLabels"},{"attributes":{"below":[{"id":"2772"}],"center":[{"id":"2775"},{"id":"2779"}],"height":300,"left":[{"id":"2776"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"renderers":[{"id":"2799"}],"sizing_mode":"fixed","title":{"id":"2764"},"toolbar":{"id":"2786"},"width":700,"x_range":{"id":"2760"},"x_scale":{"id":"2768"},"y_range":{"id":"2761"},"y_scale":{"id":"2770"}},"id":"2763","subtype":"Figure","type":"Plot"},{"attributes":{"children":[{"id":"2759"},{"id":"2763"},{"id":"2829"}],"margin":[0,0,0,0],"name":"Row05281","tags":["embedded"]},"id":"2758","type":"Row"},{"attributes":{"end":8.485310422788737,"reset_end":8.485310422788737,"reset_start":-1.5540921804637602,"start":-1.5540921804637602,"tags":[[["PC1","PC1",null]]]},"id":"2760","type":"Range1d"},{"attributes":{"end":7.83413559633714,"reset_end":7.83413559633714,"reset_start":-4.963347456561368,"start":-4.963347456561368,"tags":[[["PC2","PC2",null]]]},"id":"2761","type":"Range1d"},{"attributes":{"coordinates":null,"group":null,"text":"Contrast of Scatter plots with Original Data and PCA Data","text_color":"black","text_font_size":"12pt"},"id":"2764","type":"Title"},{"attributes":{},"id":"2809","type":"BasicTickFormatter"},{"attributes":{"callback":null,"renderers":[{"id":"2799"}],"tags":["hv_created"],"tooltips":[["PC1","@{PC1}"],["PC2","@{PC2}"],["coin_id","@{coin_id}"]]},"id":"2762","type":"HoverTool"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":0.1},"hatch_color":{"value":"#30a2da"},"line_alpha":{"value":0.1},"line_color":{"value":"#30a2da"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2797","type":"Scatter"},{"attributes":{"axis":{"id":"2772"},"coordinates":null,"grid_line_color":null,"group":null,"ticker":null},"id":"2775","type":"Grid"},{"attributes":{},"id":"2817","type":"UnionRenderers"},{"attributes":{"data":{"PC1":{"__ndarray__":"S/XQrqo447/f0e2+JFTdv0yWfGpqt9u/FGF8Nosy3r9gE49DWYbyvyLTCdRxh+C/9FpHYHTY3L8J/sd7Th7Wv5ZtTPNwyOS/2PJrm9dJ6L9L+3Og98TPv4GH9NfeDty/ihkZU4ow5r/osk+KvvmuP4JMI6evLNm/mTHFU0V66b/0BLsJMGewP7SyldMFTN+/186LH/eZ078N6SrB0W7gv9sWiTH7LNe/291vxyJW47+TEjtnb3Pav8cRKLozFNq/eGXZqrZ84z/oTdZjQtDcv4rYvTIjeOi/5LHvblTN4b9h8+bQmjXbv+hGFZE5agVAmSC8UEGl47+1vuS6vI7iv05VosuTLSBAvEmjFh7m2L90CisnUrTrP3CSuv+8lrw/RxQ7s2krE0BvXMBgQDzkv5YbmkYF++K/Dr1JmQJS3b/84a/49hDTvw==","dtype":"float64","order":"little","shape":[41]},"PC2":{"__ndarray__":"tQJY8+P36j9hP81egFfdP5TRe08qhcW/r49VGiCAzL+iRDd7ZVQAQG2kcU7LNvY/s2TwvP2Q4D9/xP1IkVfnP1ktO/eXqNs/o+HMJurAyb/LOhRjIAX2v7xmxoRtcca/R8kgkvpS3r+X9tOGdUYHQH8rtCB4sru/cKSem2Sk379tgFApNFH0vwWhJl5vcue/aENue2CC5j/UkppKWUfCv7qIS4bNlO+/VgsYmwp66j9q+cMuWpLlv7Td4x5qM8u/rBdkhXQI4j9pLiH9mlTDvzTrhbGEkuC/WW24gOcC/7/TdLLu64zfPxN2Joz3k4y/jY8PRXSt3r8M37YoLM7WvyEN8PHULA/AT0ntLg0gxT9l/tbKVRgCwOQhKIaGads/JVrQWxoSG0BDqFKDbN0AwPUO7msnAJY/VfWtmbtfwb9ISPDBzXbIvw==","dtype":"float64","order":"little","shape":[41]},"coin_id":["bitcoin","ethereum","tether","ripple","bitcoin-cash","binancecoin","chainlink","cardano","litecoin","bitcoin-cash-sv","crypto-com-chain","usd-coin","eos","monero","tron","tezos","okb","stellar","cosmos","cdai","neo","wrapped-bitcoin","leo-token","huobi-token","nem","binance-usd","iota","vechain","zcash","theta-token","dash","ethereum-classic","ethlend","maker","havven","omisego","celsius-degree-token","ontology","ftx-token","true-usd","digibyte"]},"selected":{"id":"2794"},"selection_policy":{"id":"2817"}},"id":"2793","type":"ColumnDataSource"},{"attributes":{"axis_label":"PC1","coordinates":null,"formatter":{"id":"2804"},"group":null,"major_label_policy":{"id":"2805"},"ticker":{"id":"2802"}},"id":"2772","type":"LinearAxis"},{"attributes":{"tools":[{"id":"2762"},{"id":"2780"},{"id":"2781"},{"id":"2782"},{"id":"2783"},{"id":"2784"}]},"id":"2786","type":"Toolbar"},{"attributes":{"axis":{"id":"2776"},"coordinates":null,"dimension":1,"grid_line_color":null,"group":null,"ticker":null},"id":"2779","type":"Grid"},{"attributes":{},"id":"2781","type":"PanTool"},{"attributes":{},"id":"2804","type":"BasicTickFormatter"},{"attributes":{"axis_label":"PC2","coordinates":null,"formatter":{"id":"2809"},"group":null,"major_label_policy":{"id":"2810"},"ticker":{"id":"2777"}},"id":"2776","type":"LinearAxis"},{"attributes":{},"id":"2777","type":"BasicTicker"},{"attributes":{},"id":"2782","type":"WheelZoomTool"},{"attributes":{},"id":"2794","type":"Selection"},{"attributes":{"overlay":{"id":"2785"}},"id":"2783","type":"BoxZoomTool"},{"attributes":{},"id":"2780","type":"SaveTool"},{"attributes":{"angle":{"value":0.0},"fill_alpha":{"value":1.0},"fill_color":{"value":"#30a2da"},"hatch_alpha":{"value":1.0},"hatch_color":{"value":"#30a2da"},"hatch_scale":{"value":12.0},"hatch_weight":{"value":1.0},"line_alpha":{"value":1.0},"line_cap":{"value":"butt"},"line_color":{"value":"#30a2da"},"line_dash":{"value":[]},"line_dash_offset":{"value":0},"line_join":{"value":"bevel"},"line_width":{"value":1},"marker":{"value":"circle"},"size":{"value":5.477225575051661},"x":{"field":"PC1"},"y":{"field":"PC2"}},"id":"2801","type":"Scatter"}],"root_ids":["2758"]},"title":"Bokeh Application","version":"2.4.1"}};
    var render_items = [{"docid":"4aaf8536-c5b4-479a-becc-2ab00418f977","root_ids":["2758"],"roots":{"2758":"fa84e68a-160c-4433-99de-9110fb4e7b4d"}}];
    root.Bokeh.embed.embed_items_notebook(docs_json, render_items);
  }
  if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (root.Bokeh !== undefined && root.Bokeh.Panel !== undefined) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
        }
      }
    }, 25, root)
  }
})(window);</script>




```python
df_cluster_composite.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price_change_percentage_24h</th>
      <th>price_change_percentage_7d</th>
      <th>price_change_percentage_14d</th>
      <th>price_change_percentage_30d</th>
      <th>price_change_percentage_60d</th>
      <th>price_change_percentage_200d</th>
      <th>price_change_percentage_1y</th>
      <th>CryptoCluster_x</th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>CryptoCluster_y</th>
    </tr>
    <tr>
      <th>coin_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bitcoin</th>
      <td>0.508529</td>
      <td>0.493193</td>
      <td>0.772200</td>
      <td>0.235460</td>
      <td>-0.067495</td>
      <td>-0.355953</td>
      <td>-0.251637</td>
      <td>3</td>
      <td>-0.600667</td>
      <td>0.842760</td>
      <td>0.461595</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ethereum</th>
      <td>0.185446</td>
      <td>0.934445</td>
      <td>0.558692</td>
      <td>-0.054341</td>
      <td>-0.273483</td>
      <td>-0.115759</td>
      <td>-0.199352</td>
      <td>3</td>
      <td>-0.458261</td>
      <td>0.458466</td>
      <td>0.952877</td>
      <td>0</td>
    </tr>
    <tr>
      <th>tether</th>
      <td>0.021774</td>
      <td>-0.706337</td>
      <td>-0.021680</td>
      <td>-0.061030</td>
      <td>0.008005</td>
      <td>-0.550247</td>
      <td>-0.282061</td>
      <td>0</td>
      <td>-0.433070</td>
      <td>-0.168126</td>
      <td>-0.641752</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ripple</th>
      <td>-0.040764</td>
      <td>-0.810928</td>
      <td>0.249458</td>
      <td>-0.050388</td>
      <td>-0.373164</td>
      <td>-0.458259</td>
      <td>-0.295546</td>
      <td>0</td>
      <td>-0.471835</td>
      <td>-0.222660</td>
      <td>-0.479053</td>
      <td>1</td>
    </tr>
    <tr>
      <th>bitcoin-cash</th>
      <td>1.193036</td>
      <td>2.000959</td>
      <td>1.760610</td>
      <td>0.545842</td>
      <td>-0.291203</td>
      <td>-0.499848</td>
      <td>-0.270317</td>
      <td>3</td>
      <td>-1.157800</td>
      <td>2.041209</td>
      <td>1.859715</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### Answer the following question: After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?

  * **Question:** After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?

  * **Answer:** The break is 4, so the less features used, the more it becomes centralized around the mean.
