---
title: 'Kubernetes concepts'
date: 2022-12-27T10:48:07+02:00
draft: true
math: true
images: []
description: 'Kubernetes conepts'
resources:
- name: "featured-image"
  src: "featured-image.jpg"
---

# Kubernetes concepts

* Node: server (physical or VM)
* Pod: low level abstraction over a container (abstract docker or another container technology). Usually one container by pod

* each pod has its own virtual IP address (internal to K8). if The pod die, the new one has a new Ip address.

* Service component: a permanent IP address that can be attached to each pod so pods can communicates with each others. It's also a Load balancer to distribute requests to pods less busy (which are replacted)

* External service cmponent: service visible to the outside

* Component called "ingress": to have clean URL on a service (instead of IP)

* Component: "ConfigMap": app configuration (databse Urls, etc...)

* component "Secret": configMap but to store secrets, stored in base54 encoded (passwords, certificates)

* component "Volumnes": attaches physical storage to pods (local machine or external disk). used for DB, because k8 doesn't manage any peristence

* component "Deployment": blueprint to create (and replicate) pods. only for stateless pods

* component "StatefulSet": like deployment but manage stateful pods like DB (to prevent several pods to create inconsistencies in db by concurrent read.writes). it manages read and write roles for pods. NB: it's complex, a common practice is to host database ouside the k8 cluster.


====

basic architecture:

* Nodes: "worker server": run multiple pods

* Installed on each nodes:
    * container runtime (docker)
    * kubelet: interact with the machine and the container runtime (docker)
    * kube proxy: forwarding requests to pods (smart: sending request on local pod if possible)

* "Master Nodes". several services:
    * Api Server: entry point to the cluster, cluster gateway to external admin (by UI, command Line, API); It validate requests then forward to other Master services
    * "Scheduler": when requested by user, find best node to create a new pod, the ask kubelet on the node to start the container
    * "Controller manager": detect state changes (like a pod crashing) and if needed ask scheduler to restart pods
    * "etcd": key-value store of cluster states: register all cluster states informations. distributed storage among all the master nodes.


=====

### Minikube and Kubectl

* Minikube: One Node K8s cluster on local in a virtual box. One node that run worker AND master processes. usefull on dev.

* kubectl: command line to interact with the K8 cluster (minikube or cloud). connect to APi server on master.

* Installation: need hypervisor (to run VM) to create Minikube k8 cluster (ex: hyperkit). Nb: minikube install kubeCTL also, and docker too.

===

commands kubectl

* kubectl get node
* kubectl get pod
* kubectl get services
* kubectl create deployment $NAME --image=$DOCKER_IMAGE
* kubectl get deployment
* kubectl get replicaset

replicaset: manages the replicas of a pod. admin work with deployments, replicaset is managed by k8s

layers:
deploiyment manages a replicaset, manages replica of pods, which encapsulate containers

toute la config du deployment est dans un fichier yaml

======

debugging pods:

* kubectl logs $PODNAME

displays app logs

* kubectl exec -it $PODNAME -- bin/bash: get command line into the container


create/delete etc...is done at deployement level, everything under (replica set, pods) is done automatically

=======

K8 configuration files: all deployment options

* kubectl apply -f $CONFIG_FILE.yaml

will create or update the deployment

same thing with service, volumes...

====

Config file

one component by component (deployment/service...)

3 parts to config file:

* metadata: api version, component kind
* specification: specific to component kind
* status: generated and updated

info about status comes from "etcd"

##### file format

yaml / strict indentiation -> yaml validator

thoses configs are stored with app code
("infrastructure as code")
or its own repository

pods are defined inside of deployment config (image...)

in pod config, key-value pairs (labels)

in pod metadata (inside deployment)



la commande kubectl apply will modify the config file with status (not the original but a copy)

#### complete app setup

il faut choisir l'ordre decreation des deployment s'ils se referencent les uns les autres

"---" to put several config (deployments...) in one file

#### Namespace

virtual cluster inside a k8s cluster
there is namespaces created by k8 (for master, admin...)

namespace: group ressources for clarity, separation between teams; staging VS prod, blue/green deployment...

rights restricted to namespaces by team
limited ressources by namespace

each NS must has its own configMap, idem for secret.
A service can be shared to other namespaces

some ressources are outside namespaces: ex: volumes

might not be useful for small projets=> they go to "default" namespace



#### ingress

ingress + internal service instead of external service

external service: external ip adress
ingress: routing rules: host(http adresss) and path is forwarded to service

ingres controller: implementation of ingress, ex: nginx
to evaluate the rules of the config and redirct to internal services ip

each cloud provider has ingress controller implem already done

on minikube, we can use nginx controller

#### Helm

distribute yaml files.
ex: elastic stack for logging with all deployment/services/volume...

"helm charts": bundle of yaml files
to reuse configuration




###

1 definition des terms
2) schema: en partant d'une requette web
3) config avec commentaires
4) command lines
































## What is K8s ?

## Main K8s components

## K8s Architecture