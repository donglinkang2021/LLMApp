# 使用httpx时遇到错误

```python
import httpx
client = httpx.Client()
response = client.get('http://localhost:5141')
print(response.status_code)
```

我执行上面代码时返回的结果是`502`, 但是我可以正常通过浏览器访问`http://localhost:5141`，使用requests库的get方法也可以访问网址, 为什么会出现这种情况？

```python
import requests
response = requests.get('http://localhost:5141')
print(response)
# 返回结果为 <Response [200]>
```

## 解决方法

- 关闭代理
