data = csvread('cervicalCancerFix.csv', 1, 0);
Hinselmann = data(:,29);
Schiller = data(:,30);
Cytology = data(:,31);
Biopsy = data(:,32);

data = data(:,1:28);

% centering the data
for k = 1:28
    colSum = sum(data(:,k));
    for j = 1:668
        data(j,k) = data(j,k) - (colSum/668);
    end
end

sumH = sum(Hinselmann);
sumS = sum(Schiller);
sumC = sum(Cytology);
sumB = sum(Biopsy);

for j = 1:668
    Hinselmann(j) = Hinselmann(j) - (sumH/668);
    Schiller(j) = Schiller(j) - (sumS/668);
    Cytology(j) = Cytology(j) - (sumC/668);
    Biopsy(j) = Biopsy(j) - (sumB/668);
end
[u,s,v] = svd(data);
singularValues = diag(s);

plot(singularValues, 'ko');
title('Singular Values');
ylabel('square root of eigenvalue');
xlabel('jth index');



figure(1);
a = data(:,1);
b = data(:,27);
c = data(:,9);
hold on;
grid on;
for k = 1:668
    if Hinselmann(k) == 1
        plot3(a(k), b(k), c(k), 'ro');
    else
        plot3(a(k), b(k), c(k), 'b.');
    end
end
hold off;
