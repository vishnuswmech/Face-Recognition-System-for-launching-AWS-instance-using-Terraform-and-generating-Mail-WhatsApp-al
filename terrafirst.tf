
provider "aws" {
profile= "vishnu"
region="ap-south-1"

 access_key = ""
  secret_key = ""
}

data "aws_availability_zones" "myaz11" {
  

}



resource "aws_vpc" "myterravpc1" {
cidr_block= "172.168.0.0/16"
tags= {
Name= "myterravpc"
}
}

resource "aws_internet_gateway" "myterrainternetgateway1" {

vpc_id=  aws_vpc.myterravpc1.id

tags= {
Name= "myterrainternetgateway" 
}
}




resource "aws_subnet" "myterrasubnet11" {
cidr_block= "172.168.1.0/24"
vpc_id= aws_vpc.myterravpc1.id
tags= {
Name= "myterrasubnet1"

}
}

resource "aws_route_table" "myterraroute11" {


vpc_id= aws_vpc.myterravpc1.id
route {
cidr_block= "0.0.0.0/0"
gateway_id= aws_internet_gateway.myterrainternetgateway1.id
}
tags= {
Name: "myterraroute1"
}
}

resource "aws_route_table_association" "routeassociate1" {
subnet_id= aws_subnet.myterrasubnet11.id
route_table_id= aws_route_table.myterraroute11.id
}



resource "aws_subnet" "myterrasubnet12" {
cidr_block= "172.168.2.0/24"
vpc_id= aws_vpc.myterravpc1.id
availability_zone=data.aws_availability_zones.myaz11.names[0]
tags= {
Name= "myterrasubnet2"
}
}

resource "aws_security_group" "myterrasg1" {

name= "myterrasg"
vpc_id= aws_vpc.myterravpc1.id
ingress {
from_port= 0
to_port= 0
protocol= "-1"
 cidr_blocks      = ["0.0.0.0/0"]
}

egress {
from_port= 0
to_port= 0
protocol= "-1"
cidr_blocks= [aws_vpc.myterravpc1.cidr_block]
}
} 


resource "aws_instance" "myterrainstance11" {
ami= "ami-010aff33ed5991201"
instance_type= "t2.micro"


associate_public_ip_address = "true"
vpc_security_group_ids= [aws_security_group.myterrasg1.id]
subnet_id= aws_subnet.myterrasubnet11.id
availability_zone= data.aws_availability_zones.myaz11.names[0]
key_name= "Multikeypair"
tags= {
Name= "myterrainstance1"
}
}

resource "aws_ebs_volume" "myebs" {
  availability_zone = data.aws_availability_zones.myaz11.names[0]
  size              = 5

  tags = {
    Name = "MyEBS"
  }
}
resource "aws_volume_attachment" "ebs_attach" {
  device_name = "/dev/sds"
  volume_id   = aws_ebs_volume.myebs.id
  instance_id = aws_instance.myterrainstance11.id
}

output "out" {

value= "${aws_instance.myterrainstance11.public_ip}"

}