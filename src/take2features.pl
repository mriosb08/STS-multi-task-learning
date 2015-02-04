#!/usr/bin/perl
use File::Basename;
#modifies svm fortmat for input of multi-task GP
#adds tasd if for each training file used
my $task_id = 1;
foreach my $name(@ARGV){
    open(my $FILE, $name) or die "file not found\n";
    my($filename, $directories, $suffix) = fileparse($name);
    my $i = 0;
    while(my $line = <$FILE>){
        chomp($line);
        $line =~ s/[0-9]+://g;
        $line =~ s/NaN/0.0/g;
        print "$task_id|||$filename|||$i $line\n";
        $i++;       
    }
    $task_id++;
}
