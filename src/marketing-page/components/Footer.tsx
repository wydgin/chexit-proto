import Box from '@mui/material/Box';
import Container from '@mui/material/Container';
import Link from '@mui/material/Link';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import { Link as RouterLink } from 'react-router-dom';

/**
 * Slim site footer. The previous newsletter + Product / Company / Legal columns
 * were removed in favor of just a copyright line and an About-us link.
 */
export default function Footer() {
  return (
    <Box
      component="footer"
      sx={{
        borderTop: '1px solid',
        borderColor: 'divider',
        mt: 4,
      }}
    >
      <Container
        sx={{
          display: 'flex',
          flexDirection: { xs: 'column', sm: 'row' },
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: 1,
          py: 3,
        }}
      >
        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
          © {new Date().getFullYear()} Chexit
        </Typography>
        <Stack direction="row" spacing={3}>
          <Link
            component={RouterLink}
            to="/about"
            color="text.secondary"
            variant="body2"
            underline="hover"
          >
            About us
          </Link>
        </Stack>
      </Container>
    </Box>
  );
}
