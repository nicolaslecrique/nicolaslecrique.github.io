---
title: "My First Post"
date: 2022-07-15T10:48:07+02:00
draft: false
math: true
---



# How are you

fine

## Subtitle

finefine

### Subtitle 2

 Some content

#### subtitle 3

**gras**
*italic*
~~barré~~
__lol__

>retrait
>retrait2


```python
def fix_hostname(hostname):
    hostname = re.sub(r'[\\/:"*?<>|]+', "-", hostname)
    hostname = hostname.lower()
    return hostname

hostname = fix_hostname(device["name"])
```


 ```js

func GetTitleFunc(style string) func(s string) string {
  switch strings.ToLower(style) {
  case "go":
    return strings.Title
  case "chicago":
    return transform.NewTitleConverter(transform.ChicagoStyle)
  default:
    return transform.NewTitleConverter(transform.APStyle)
  }
}

 ```


$\int_{-\infty}^{\infty} e^{-x^2} dx$.

$$\int_{-\infty}^{\infty} e^{-x^2} dx$$.