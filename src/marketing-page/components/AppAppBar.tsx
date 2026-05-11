import * as React from 'react';
import { styled, alpha } from '@mui/material/styles';
import Box from '@mui/material/Box';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton';
import Container from '@mui/material/Container';
import Divider from '@mui/material/Divider';
import MenuItem from '@mui/material/MenuItem';
import Drawer from '@mui/material/Drawer';
import MenuIcon from '@mui/icons-material/Menu';
import CloseRoundedIcon from '@mui/icons-material/CloseRounded';
import { Link as RouterLink, useLocation, useNavigate } from 'react-router-dom';
import ColorModeIconDropdown from '../../../shared-theme/ColorModeIconDropdown';
import Sitemark from './SitemarkIcon';

const StyledToolbar = styled(Toolbar)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  flexShrink: 0,
  borderRadius: `calc(${theme.shape.borderRadius}px + 8px)`,
  backdropFilter: 'blur(24px)',
  border: '1px solid',
  borderColor: (theme.vars || theme).palette.divider,
  backgroundColor: theme.vars
    ? `rgba(${theme.vars.palette.background.defaultChannel} / 0.4)`
    : alpha(theme.palette.background.default, 0.4),
  boxShadow: (theme.vars || theme).shadows[1],
  padding: '8px 12px',
}));

/** Pretty top-nav link: refined type, subtle hover pill, active-state highlight. */
function NavLink({
  to,
  active,
  children,
}: {
  to: string;
  active: boolean;
  children: React.ReactNode;
}) {
  return (
    <Button
      component={RouterLink}
      to={to}
      disableRipple
      sx={(theme) => ({
        position: 'relative',
        px: 1.5,
        py: 0.5,
        minWidth: 0,
        borderRadius: 999,
        fontSize: '0.875rem',
        fontWeight: active ? 600 : 500,
        lineHeight: 1.4,
        letterSpacing: '0.01em',
        textTransform: 'none',
        color: active ? 'text.primary' : 'text.secondary',
        backgroundColor: 'transparent',
        transition:
          'color 120ms ease, background-color 120ms ease, transform 120ms ease',
        '&:hover': {
          color: 'text.primary',
          backgroundColor: alpha(theme.palette.text.primary, 0.06),
        },
        '&:active': {
          transform: 'scale(0.98)',
        },
        ...(active && {
          '&::after': {
            content: '""',
            position: 'absolute',
            left: '50%',
            bottom: 2,
            transform: 'translateX(-50%)',
            width: 16,
            height: 2,
            borderRadius: 2,
            backgroundColor: 'primary.main',
            ...theme.applyStyles('dark', {
              backgroundColor: theme.palette.primary.light,
            }),
          },
        }),
      })}
    >
      {children}
    </Button>
  );
}

export default function AppAppBar() {
  const [open, setOpen] = React.useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const currentPath = location.pathname;

  const toggleDrawer = (newOpen: boolean) => () => {
    setOpen(newOpen);
  };

  const goTo = (path: string) => () => {
    setOpen(false);
    navigate(path);
  };

  const navItems = [
    { label: 'Diagnosis Dashboard', to: '/dashboard' },
    { label: 'About us', to: '/about' },
  ];

  return (
    <AppBar
      position="fixed"
      enableColorOnDark
      sx={{
        boxShadow: 0,
        bgcolor: 'transparent',
        backgroundImage: 'none',
        mt: 'calc(var(--template-frame-height, 0px) + 28px)',
      }}
    >
      <Container maxWidth="lg">
        <StyledToolbar variant="dense" disableGutters>
          <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', px: 0 }}>
            <Sitemark />
            <Box
              component="nav"
              aria-label="Primary"
              sx={{
                display: { xs: 'none', md: 'flex' },
                alignItems: 'center',
                gap: 0.25,
                ml: 1.5,
              }}
            >
              {navItems.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  active={currentPath === item.to}
                >
                  {item.label}
                </NavLink>
              ))}
            </Box>
          </Box>
          <Box
            sx={{
              display: { xs: 'none', md: 'flex' },
              gap: 1,
              alignItems: 'center',
            }}
          >
            <ColorModeIconDropdown />
          </Box>
          <Box sx={{ display: { xs: 'flex', md: 'none' }, gap: 1 }}>
            <ColorModeIconDropdown size="medium" />
            <IconButton aria-label="Menu button" onClick={toggleDrawer(true)}>
              <MenuIcon />
            </IconButton>
            <Drawer
              anchor="top"
              open={open}
              onClose={toggleDrawer(false)}
              PaperProps={{
                sx: {
                  top: 'var(--template-frame-height, 0px)',
                },
              }}
            >
              <Box sx={{ p: 2, backgroundColor: 'background.default' }}>
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'flex-end',
                  }}
                >
                  <IconButton onClick={toggleDrawer(false)}>
                    <CloseRoundedIcon />
                  </IconButton>
                </Box>

                {navItems.map((item) => {
                  const active = currentPath === item.to;
                  return (
                    <MenuItem
                      key={item.to}
                      onClick={goTo(item.to)}
                      selected={active}
                      sx={{
                        borderRadius: 1.5,
                        my: 0.25,
                        py: 1,
                        fontSize: '0.95rem',
                        fontWeight: active ? 600 : 500,
                        color: active ? 'text.primary' : 'text.secondary',
                      }}
                    >
                      {item.label}
                    </MenuItem>
                  );
                })}
                <Divider sx={{ my: 3 }} />
                <MenuItem>
                  <ColorModeIconDropdown size="medium" />
                </MenuItem>
              </Box>
            </Drawer>
          </Box>
        </StyledToolbar>
      </Container>
    </AppBar>
  );
}
