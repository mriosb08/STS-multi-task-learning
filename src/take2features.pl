#!/usr/bin/perl

while(my $line = <STDIN>){
	$line =~ s/[0-9]+://g;
	($y, @x_tmp) = split(/\s+/, $line);
	$x = join('|||', @x_tmp);
	print "$y\t$x\n";
}
